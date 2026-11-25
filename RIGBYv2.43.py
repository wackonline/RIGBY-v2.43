#!/usr/bin/env python3
"""
rigby_exec.py - RIGBY v2.43 (alpha) - LIVE enforcement + execution layer (Solana).

The LLM proposes. This file decides and trades. All limits live in CONFIG;
LLM instruction fields can only tighten them, never loosen.

Venue:  Jupiter Swap API V2 (GET /order, POST /execute). Base asset: USDC.
        Wallet needs USDC (trading balance) + a little SOL (fees/rent).

Env:    SOLANA_RPC      paid RPC URL
        JUP_API_KEY     key from developers.jup.ag/portal
        RIGBY_KEYFILE   path to Solana keypair JSON (64-int array), chmod 600,
                        dedicated hot wallet. Key is never logged or put in env.

Commands:
  python rigby_exec.py check        preflight: env, key, RPC, Jupiter key, balances
  python rigby_exec.py limits       prints EQUITY_USD / MAX_SIZE_USD for the LLM cycle input
  python rigby_exec.py submit       reads LLM output (or bare JSON) on stdin, runs execute()
  python rigby_exec.py loop         runs monitor() on a timer (keep running)
  python rigby_exec.py report       expectancy stats from the trade journal (real fills)
  python rigby_exec.py status
  python rigby_exec.py clear-halt   owner-only: clears a manual halt
Library: from rigby_exec import execute, monitor

Run execute() and monitor() from the same host. A lock file serialises access
to the state file.
"""
import base64
import json
import math
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import requests

CONFIG = {
    "max_pct": 0.015,                 # max position as fraction of equity
    "max_open": 1,
    "daily_loss_pct": 0.08,
    "consec_loss_halt": 2,
    "halt_hours": 12,
    "stop_pct": -10.0,
    "trail_pct": 20.0,
    "time_stop_min": 20,
    "time_stop_min_gain_pct": 10.0,
    "cooldown_min": 20,
    "slippage_cap_pct": 1.5,          # entry slippage + max price impact
    "exit_slippage_ladder_bps": [300, 1000, 3000],  # exits must land
    "max_round_trip_loss_pct": 5.0,   # buy->sell quote loss allowed at entry
    "max_instr_age_s": 60,
    "default_tp1": 2.0,
    "default_tp2": 3.0,
    "min_sol_lamports": 10_000_000,   # 0.01 SOL for fees/rent
    "monitor_interval_s": 5,
}

RPC_URL = os.getenv("SOLANA_RPC", "")
JUP_BASE = "https://api.jup.ag/swap/v2"
JUP_KEY = os.getenv("JUP_API_KEY", "")
KEYFILE = os.getenv("RIGBY_KEYFILE", "")
STATE_FILE = os.getenv("RIGBY_STATE", "rigby_state.json")
LOG_FILE = os.getenv("RIGBY_LOG", "rigby.log")
JOURNAL = os.getenv("RIGBY_JOURNAL", "trades.jsonl")

USDC = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDC_DEC = 6
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"  # Token-2022 rejected
ALLOWED_KEYS = {"action", "token", "size_usd", "stop_pct", "tp1", "tp2", "time_stop_min"}
BLOCKED_MINTS = {
    "So11111111111111111111111111111111111111112",
    USDC,
    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
}
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


# ---------- helpers ----------
def now():
    return datetime.now(timezone.utc)


def iso(t):
    return t.isoformat()


def parse(s):
    return datetime.fromisoformat(s)


def log(msg):
    line = f"[{iso(now())}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def journal(event, **kw):
    """Append-only trade journal (actual fills). Feeds `report`."""
    with open(JOURNAL, "a") as f:
        f.write(json.dumps({"ts": iso(now()), "event": event, **kw}) + "\n")


def finite_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def valid_pubkey(s):
    """True only for base58 that decodes to exactly 32 bytes."""
    if not isinstance(s, str) or not 32 <= len(s) <= 44:
        return False
    n = 0
    for c in s:
        i = B58.find(c)
        if i < 0:
            return False
        n = n * 58 + i
    pad = len(s) - len(s.lstrip("1"))
    return pad + (n.bit_length() + 7) // 8 == 32


@contextmanager
def state_lock(timeout=30):
    lock = STATE_FILE + ".lock"
    t0 = time.time()
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > 300:  # stale
                    os.remove(lock)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise RuntimeError("state lock timeout")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


# ---------- network ----------
def _require_env():
    if not (RPC_URL and JUP_KEY and KEYFILE):
        raise RuntimeError("set SOLANA_RPC, JUP_API_KEY, RIGBY_KEYFILE")


def rpc(method, params):
    r = requests.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": method,
                                     "params": params}, timeout=10)
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"rpc {method}: {j['error']}")
    return j["result"]


def _jup_get(path, params):
    r = requests.get(f"{JUP_BASE}{path}", params=params,
                     headers={"x-api-key": JUP_KEY}, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"jup GET {path} {r.status_code}: {r.text[:200]}")
    return r.json()


def _jup_post(path, body):
    r = requests.post(f"{JUP_BASE}{path}", json=body,
                      headers={"x-api-key": JUP_KEY}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"jup POST {path} {r.status_code}: {r.text[:200]}")
    return r.json()


_KP = None


def keypair():
    global _KP
    if _KP is None:
        from solders.keypair import Keypair
        if os.stat(KEYFILE).st_mode & 0o077:
            raise RuntimeError("keyfile must be chmod 600")
        with open(KEYFILE) as f:
            _KP = Keypair.from_bytes(bytes(json.load(f)))
    return _KP


def wallet():
    return str(keypair().pubkey())


def sign_b64(tx_b64):
    """Sign only our slot; keeps any other signatures already on the tx."""
    from solders.message import to_bytes_versioned
    from solders.transaction import VersionedTransaction
    kp = keypair()
    tx = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
    msg = tx.message
    keys = list(msg.account_keys)
    me = kp.pubkey()
    if me not in keys[:msg.header.num_required_signatures]:
        raise RuntimeError("wallet is not a required signer")
    sigs = list(tx.signatures)
    sigs[keys.index(me)] = kp.sign_message(to_bytes_versioned(msg))
    return base64.b64encode(bytes(VersionedTransaction.populate(msg, sigs))).decode()


def jup_order(in_mint, out_mint, amount_raw, slippage_bps, taker=False):
    p = {"inputMint": in_mint, "outputMint": out_mint,
         "amount": str(int(amount_raw)), "slippageBps": int(slippage_bps)}
    if taker:
        p["taker"] = wallet()
    return _jup_get("/order", p)


def jup_execute(order):
    body = {"signedTransaction": sign_b64(order["transaction"]),
            "requestId": order["requestId"]}
    if order.get("lastValidBlockHeight"):
        body["lastValidBlockHeight"] = order["lastValidBlockHeight"]
    res = _jup_post("/execute", body)
    if res.get("status") != "Success":
        raise RuntimeError(f"execute failed: code={res.get('code')} {res.get('error')}")
    return res


def token_balance_raw(mint):
    r = rpc("getTokenAccountsByOwner", [wallet(), {"mint": mint}, {"encoding": "jsonParsed"}])
    return sum(int(a["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
               for a in r["value"])


def get_equity_usd(st):
    """USDC balance + cost basis of open positions."""
    usdc = token_balance_raw(USDC) / 10 ** USDC_DEC
    return usdc + sum(p["cost_usd"] for p in st["open_positions"])


# ---------- gates ----------
def mint_check(token):
    """Returns decimals if safe, else None. Requires: classic SPL token program,
    mint authority renounced, freeze authority renounced."""
    v = rpc("getAccountInfo", [token, {"encoding": "jsonParsed"}])["value"]
    if not v or v.get("owner") != TOKEN_PROGRAM:
        return None
    d = v.get("data")
    if not isinstance(d, dict) or d.get("parsed", {}).get("type") != "mint":
        return None
    info = d["parsed"]["info"]
    if info.get("mintAuthority") is not None or info.get("freezeAuthority") is not None:
        return None
    return int(info["decimals"])


def confirm_sell_route(token, out_amount_raw, cost_usd):
    """Quote the sell of what we'd receive. Needs a route, low price impact,
    and a bounded round-trip loss."""
    try:
        q = jup_order(token, USDC, out_amount_raw,
                      CONFIG["slippage_cap_pct"] * 100, taker=False)
        out = int(q.get("outAmount") or 0)
        if out <= 0:
            return False
        if abs(float(q.get("priceImpact") or 0)) > CONFIG["slippage_cap_pct"]:
            return False
        return out / 10 ** USDC_DEC >= cost_usd * (1 - CONFIG["max_round_trip_loss_pct"] / 100)
    except Exception as e:
        log(f"sell-route check error: {e}")
        return False


# ---------- state ----------
def load_state():
    try:
        with open(STATE_FILE) as f:
            st = json.load(f)
    except FileNotFoundError:
        st = {"day": str(now().date()), "day_start_equity": None, "daily_pnl_usd": 0.0,
              "consecutive_losses": 0, "halt_until": None, "manual_halt": None,
              "open_positions": [], "last_close": {}}
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"state unreadable, refusing to run: {e}")
    today = str(now().date())
    if st["day"] != today:
        st["day"], st["daily_pnl_usd"], st["day_start_equity"] = today, 0.0, None
    return st


def save_state(st):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(STATE_FILE)))
    with os.fdopen(fd, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE_FILE)


def manual_halt(st, reason):
    st["manual_halt"] = reason
    save_state(st)
    log(f"MANUAL HALT: {reason} (owner must run: python rigby_exec.py clear-halt)")


def is_halted(st):
    if st.get("manual_halt"):
        return True
    t = now()
    if st["halt_until"] and t < parse(st["halt_until"]):
        return True
    dse = st["day_start_equity"]
    if (dse is not None and st["daily_pnl_usd"] <= -dse * CONFIG["daily_loss_pct"]) \
            or st["consecutive_losses"] >= CONFIG["consec_loss_halt"]:
        st["halt_until"] = iso(t + timedelta(hours=CONFIG["halt_hours"]))
        st["consecutive_losses"] = 0
        log("HALT TRIGGERED")
        save_state(st)
        return True
    return False


# ---------- entry ----------
def _rej(msg):
    log(f"REJECT: {msg}")
    return False


def execute(instr, generated_at=None):
    with state_lock():
        return _execute(instr, generated_at)


def _execute(instr, generated_at):
    _require_env()
    st = load_state()
    if is_halted(st):
        return _rej("halted")

    if not isinstance(instr, dict):
        return _rej("instruction not an object")
    extra = set(instr) - ALLOWED_KEYS
    if extra:
        return _rej(f"unexpected keys {sorted(extra)}")
    if instr.get("action") != "BUY":
        return _rej("action must be BUY")
    token = instr.get("token")
    if not valid_pubkey(token):
        return _rej("invalid mint address")
    if token in BLOCKED_MINTS:
        return _rej("blocked mint")
    size = instr.get("size_usd")
    if not finite_num(size):
        return _rej(f"size not a finite number: {size!r}")
    size = float(size)
    if generated_at is not None and (now() - generated_at).total_seconds() > CONFIG["max_instr_age_s"]:
        return _rej("stale instruction")

    try:
        equity = get_equity_usd(st)
        sol = rpc("getBalance", [wallet()])["value"]
    except Exception as e:
        return _rej(f"wallet state unavailable: {e}")
    if sol < CONFIG["min_sol_lamports"]:
        return _rej("insufficient SOL for fees")
    if st["day_start_equity"] is None:
        st["day_start_equity"] = equity
        save_state(st)
    cap = equity * CONFIG["max_pct"]
    if not (0 < size <= cap):
        return _rej(f"size {size} outside (0, {cap:.4f}]")

    if len(st["open_positions"]) >= CONFIG["max_open"]:
        return _rej("max positions")
    if any(p["token"] == token for p in st["open_positions"]):
        return _rej("already holding token")
    lc = st["last_close"].get(token)
    if lc and (now() - parse(lc)).total_seconds() < CONFIG["cooldown_min"] * 60:
        return _rej("cooldown")

    try:
        decimals = mint_check(token)
    except Exception as e:
        return _rej(f"mint check error: {e}")
    if decimals is None:
        return _rej("mint check failed (authorities, program, or unreadable)")

    try:
        order = jup_order(USDC, token, round(size * 10 ** USDC_DEC),
                          CONFIG["slippage_cap_pct"] * 100, taker=True)
    except Exception as e:
        return _rej(f"order failed: {e}")
    if not order.get("transaction"):
        return _rej(f"no transaction: code={order.get('errorCode')} {order.get('errorMessage')}")
    if abs(float(order.get("priceImpact") or 0)) > CONFIG["slippage_cap_pct"]:
        return _rej("price impact over cap")
    out_raw = int(order.get("outAmount") or 0)
    if out_raw <= 0:
        return _rej("zero output")
    if not confirm_sell_route(token, out_raw, size):
        return _rej("sell route not confirmed")

    tp1, tp2 = instr.get("tp1"), instr.get("tp2")
    if not (finite_num(tp1) and finite_num(tp2) and 1 < tp1 < tp2):
        tp1, tp2 = CONFIG["default_tp1"], CONFIG["default_tp2"]

    try:
        res = jup_execute(order)
        cost = int(res["totalInputAmount"]) / 10 ** USDC_DEC
        got = int(res["totalOutputAmount"])
    except Exception as e:
        log(f"buy outcome uncertain: {e}")
        try:
            got = token_balance_raw(token)
        except Exception as e2:
            manual_halt(st, f"buy outcome unknown for {token}: {e2}")
            return False
        if got <= 0:
            return _rej("buy did not land")
        cost = size
        log("CRITICAL: buy landed despite error; position recovered from chain balance")

    st["open_positions"].append({
        "token": token, "decimals": decimals, "token_raw": got, "cost_usd": cost,
        "peak_value_usd": cost, "opened_at": iso(now()), "tp1": tp1, "tp2": tp2,
        "tp1_hit": False, "realized_usd": 0.0,
    })
    save_state(st)
    log(f"BUY {token} cost=${cost:.4f} tokens_raw={got}")
    journal("BUY", token=token, cost_usd=cost, tokens_raw=got, equity_usd=equity)
    return True


# ---------- exits ----------
def sell(st, pos, fraction, reason):
    amount = pos["token_raw"] if fraction >= 1.0 else int(pos["token_raw"] * fraction)
    if amount <= 0:
        return False
    for bps in CONFIG["exit_slippage_ladder_bps"]:
        try:
            res = jup_execute(jup_order(pos["token"], USDC, amount, bps, taker=True))
            got = int(res["totalOutputAmount"]) / 10 ** USDC_DEC
            cost_sold = pos["cost_usd"] * amount / pos["token_raw"]
            pnl = got - cost_sold
            pos["token_raw"] -= amount
            pos["cost_usd"] -= cost_sold
            pos["realized_usd"] += pnl
            st["daily_pnl_usd"] += pnl
            if pos["token_raw"] <= 0:
                st["consecutive_losses"] = st["consecutive_losses"] + 1 if pos["realized_usd"] < 0 else 0
                st["last_close"][pos["token"]] = iso(now())
            log(f"EXIT {reason} {pos['token']} frac={fraction} slip={bps}bps pnl=${pnl:.4f}")
            journal("EXIT", token=pos["token"], reason=reason, proceeds_usd=got,
                    cost_sold_usd=cost_sold, pnl_usd=pnl, slippage_bps=bps,
                    closed=pos["token_raw"] <= 0)
            return True
        except Exception as e:
            log(f"sell attempt at {bps}bps failed: {e}")
    manual_halt(st, f"CRITICAL: could not sell {pos['token']} ({reason})")
    return False


def monitor():
    with state_lock():
        _monitor()


def _monitor():
    _require_env()
    st = load_state()
    t = now()
    for pos in list(st["open_positions"]):
        age_min = (t - parse(pos["opened_at"])).total_seconds() / 60
        bal = None
        try:
            bal = token_balance_raw(pos["token"])
        except Exception as e:
            log(f"WARN balance check failed: {e}")
        if bal is not None and bal < pos["token_raw"] * 0.99:
            manual_halt(st, f"reconcile mismatch {pos['token']}: chain={bal} recorded={pos['token_raw']}")
            if bal <= 0:  # position gone: book as total loss, stop managing it
                st["daily_pnl_usd"] -= pos["cost_usd"]
                st["consecutive_losses"] += 1
                st["last_close"][pos["token"]] = iso(now())
                journal("EXIT", token=pos["token"], reason="RECONCILE_ZERO", proceeds_usd=0.0,
                        cost_sold_usd=pos["cost_usd"], pnl_usd=-pos["cost_usd"], closed=True)
                pos["token_raw"] = 0
                continue
            ratio = bal / pos["token_raw"]  # keep managing what's really held
            pos["cost_usd"] *= ratio
            pos["token_raw"] = bal

        try:
            q = jup_order(pos["token"], USDC, pos["token_raw"],
                          CONFIG["slippage_cap_pct"] * 100, taker=False)
            value = int(q.get("outAmount") or 0) / 10 ** USDC_DEC
            if value <= 0:
                raise RuntimeError("zero exit value")
        except Exception as e:
            log(f"WARN no exit quote for {pos['token']}: {e}")
            if age_min >= CONFIG["time_stop_min"]:
                sell(st, pos, 1.0, "NO_QUOTE_TIMEOUT")
            continue

        cost = pos["cost_usd"]
        pos["peak_value_usd"] = max(pos["peak_value_usd"], value)
        peak = pos["peak_value_usd"]
        pnl_pct = (value / cost - 1) * 100

        reason = None
        if pnl_pct <= CONFIG["stop_pct"]:
            reason = "HARD_STOP"
        elif peak > cost and (peak - value) / peak * 100 >= CONFIG["trail_pct"]:
            reason = "TRAIL"
        elif value / cost >= pos["tp2"]:
            reason = "TP2"
        elif age_min >= CONFIG["time_stop_min"] and pnl_pct < CONFIG["time_stop_min_gain_pct"]:
            reason = "TIME_STOP"

        if reason:
            sell(st, pos, 1.0, reason)
        elif value / cost >= pos["tp1"] and not pos["tp1_hit"]:
            if sell(st, pos, 0.5, "TP1_PARTIAL"):
                pos["tp1_hit"] = True
                pos["peak_value_usd"] *= 0.5
    st["open_positions"] = [p for p in st["open_positions"] if p["token_raw"] > 0]
    save_state(st)


def report():
    """Expectancy from real closed trades (partials are folded into their trade).
    pnl = USDC out minus USDC in at actual fills; excludes SOL network/priority
    fees and token-account rent."""
    try:
        with open(JOURNAL) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        rows = []
    cur, trades = {}, []
    for r in rows:
        if r["event"] == "EXIT":
            cur[r["token"]] = cur.get(r["token"], 0.0) + r["pnl_usd"]
            if r.get("closed"):
                trades.append(cur.pop(r["token"]))
    n = len(trades)
    if n == 0:
        print("No closed trades yet.")
        return
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gross_l = -sum(losses)
    print(f"Closed trades: {n} | win rate: {len(wins) / n:.0%}")
    print(f"Total net: ${sum(trades):.4f} | expectancy/trade: ${sum(trades) / n:.4f}")
    print(f"Avg win: ${(sum(wins) / len(wins)) if wins else 0:.4f} | "
          f"avg loss: ${(-gross_l / len(losses)) if losses else 0:.4f}")
    print(f"Profit factor: {(sum(wins) / gross_l) if gross_l else float('inf'):.2f} | "
          f"best: ${max(trades):.4f} | worst: ${min(trades):.4f}")


def limits():
    st = load_state()
    eq = get_equity_usd(st)
    return {"EQUITY_USD": round(eq, 4), "MAX_SIZE_USD": round(eq * CONFIG["max_pct"], 4)}


def preflight():
    _require_env()
    SOL = "So11111111111111111111111111111111111111112"
    checks = {}

    def step(name, fn):
        try:
            v = fn()
            log(f"OK   {name}: {v}")
            checks[name] = v
            return True
        except Exception as e:
            log(f"FAIL {name}: {e}")
            return False

    ok = all([
        step("wallet", wallet),
        step("SOL balance", lambda: rpc("getBalance", [wallet()])["value"] / 1e9),
        step("USDC balance", lambda: token_balance_raw(USDC) / 10 ** USDC_DEC),
        step("jupiter key", lambda: jup_order(USDC, SOL, 10 ** USDC_DEC, 50)["outAmount"]),
    ])
    if ok:
        if checks["SOL balance"] * 1e9 < CONFIG["min_sol_lamports"]:
            log("FAIL SOL balance below minimum for fees")
            ok = False
        if checks["USDC balance"] <= 0:
            log("FAIL no USDC to trade with")
            ok = False
    if ok:
        log(f"READY {limits()}")
    return ok


def parse_instruction(text):
    for line in text.splitlines():
        if line.strip().startswith("INSTRUCTION:"):
            body = line.split(":", 1)[1].strip().strip("` ")
            return None if body.lower().startswith("none") else json.loads(body)
    return json.loads(text)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "loop":
        log("RIGBY monitor loop started")
        while True:
            try:
                monitor()
            except Exception as e:
                log(f"monitor error: {e}")
            time.sleep(CONFIG["monitor_interval_s"])
    elif cmd == "check":
        sys.exit(0 if preflight() else 1)
    elif cmd == "report":
        report()
    elif cmd == "limits":
        print(json.dumps(limits()))
    elif cmd == "submit":
        try:
            instr = parse_instruction(sys.stdin.read())
        except Exception as e:
            log(f"REJECT: unparseable instruction: {e}")
            sys.exit(1)
        if instr is None:
            log("SKIP: no instruction")
            sys.exit(0)
        sys.exit(0 if execute(instr, generated_at=now()) else 1)
    elif cmd == "clear-halt":
        with state_lock():
            s = load_state()
            s["manual_halt"] = None
            save_state(s)
            log("manual halt cleared by owner")
    else:
        s = load_state()
        log(f"open={len(s['open_positions'])} daily_pnl={s['daily_pnl_usd']:.4f} "
            f"manual_halt={s['manual_halt']} halted={is_halted(s)}")
