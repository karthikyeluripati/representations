"""WebState: a minimal executable web-app simulator for false-novelty / false-equivalence studies.

Pure Python, deterministic, no browser. Each *family* is a small multi-page app whose
observations are accessibility-tree-like text. Some state is hidden (never rendered directly).

Semantics
- `state` is a dict of all variables, including hidden ones (privileged; used only for labels).
- `sem_obs(state)` is the render-independent observation: (page, ((eid, role, value), ...), toast).
- `render(sem_obs, cfg)` produces the text the agent sees. `cfg` controls cosmetic perturbations:
  element order (layout), label synonyms (rename), irrelevant static text (decor), style attributes.
- `step(state, eid)` executes a click on element `eid` and returns a new state.
- Behavioral equivalence over horizon H (`h_equivalent`) compares the render-independent observation
  sequences produced by *every* action sequence of length <= H from the two states. This is
  H-step trace equivalence of the deterministic system, verified by execution.

Deployment-side methods only ever see rendered text (goal, observations, action labels).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

CLICK_ROLES = ("button", "link", "checkbox")


def h(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# Families. Each defines: LABELS (eid -> 3 label variants), TITLES (page -> 3 variants),
# HIDDEN (name -> choices), COMMITTAL (eids with external side effects), init(hidden),
# visible(state) -> list[(eid, role, value)], transition(state, eid) -> state (toast already cleared).
# ----------------------------------------------------------------------------------------------


class Family:
    name = "base"
    GOAL = ""
    LABELS: dict = {}
    TITLES: dict = {}
    HIDDEN: dict = {}
    COMMITTAL: set = set()

    def init(self, hidden):  # pragma: no cover - abstract
        raise NotImplementedError

    def visible(self, s):  # pragma: no cover
        raise NotImplementedError

    def transition(self, s, eid):  # pragma: no cover
        raise NotImplementedError

    def tick(self, s):
        """Time-driven dynamics applied after every action (delayed effects)."""
        return s


class Settings(Family):
    name = "settings"
    GOAL = "Turn on notifications and keep the change."
    LABELS = {
        "settings_link": ["Settings", "Preferences", "Options"],
        "account_link": ["Account", "Profile", "My account"],
        "menu_btn": ["Menu", "Navigation", "More"],
        "logout_btn": ["Log out", "Sign out", "Exit"],
        "home_link": ["Home", "Back to home", "Main page"],
        "notif": ["Email notifications", "Notify me by email", "Email alerts"],
        "save_btn": ["Save", "Apply", "Store changes"],
        "role_field": ["Role", "Access level", "Permission"],
        "delete_btn": ["Delete account", "Remove account", "Close account"],
        "confirm_text": ["Discard unsaved changes?", "Leave without saving?", "Changes will be lost. Continue?"],
        "confirm_yes": ["Yes", "Discard", "Continue"],
        "confirm_no": ["No", "Stay", "Cancel"],
        "status": ["Status", "State", "Session"],
    }
    TITLES = {
        "home": ["Home", "Dashboard", "Start"],
        "menu": ["Menu", "Navigation", "Site map"],
        "settings": ["Settings", "Preferences", "Options"],
        "account": ["Account", "Profile", "My account"],
        "confirm": ["Confirm", "Are you sure", "Confirmation"],
        "bye": ["Signed out", "Goodbye", "Logged out"],
        "deleted": ["Account removed", "Deleted", "Closed"],
    }
    HIDDEN = {"role": ["user", "admin"], "quota_full": [False, True]}
    COMMITTAL = {"logout_btn", "confirm_yes", "delete_btn", "save_btn"}

    def init(self, hidden):
        return {"page": "home", "toast": None, "notif_saved": False, "notif_draft": False, **hidden}

    def visible(self, s):
        p = s["page"]
        if p == "home":
            return [("settings_link", "link", None), ("account_link", "link", None),
                    ("menu_btn", "button", None), ("logout_btn", "button", None)]
        if p == "menu":
            return [("settings_link", "link", None), ("home_link", "link", None)]
        if p == "settings":
            return [("notif", "checkbox", "on" if s["notif_draft"] else "off"),
                    ("save_btn", "button", None), ("home_link", "link", None)]
        if p == "account":
            return [("role_field", "field", s["role"]), ("delete_btn", "button", None),
                    ("home_link", "link", None)]
        if p == "confirm":
            return [("confirm_text", "field", None), ("confirm_yes", "button", None),
                    ("confirm_no", "button", None)]
        if p == "bye":
            return [("status", "field", "notifications " + ("on" if s["notif_saved"] else "off"))]
        if p == "deleted":
            return [("status", "field", "deleted")]
        raise ValueError(p)

    def transition(self, s, e):
        if e in ("settings_link",):
            s["page"] = "settings"
        elif e == "account_link":
            s["page"] = "account"
        elif e == "menu_btn":
            s["page"] = "menu"
        elif e == "home_link" or e == "confirm_no":
            s["page"] = "home"
        elif e == "notif":
            s["notif_draft"] = not s["notif_draft"]
        elif e == "save_btn":
            if s["quota_full"]:
                s["toast"] = "Save failed"
            else:
                s["notif_saved"] = s["notif_draft"]
                s["toast"] = "Saved"
        elif e == "logout_btn":
            s["page"] = "confirm" if s["notif_draft"] != s["notif_saved"] else "bye"
        elif e == "confirm_yes":
            s["notif_draft"] = s["notif_saved"]
            s["page"] = "bye"
        elif e == "delete_btn":
            if s["role"] == "admin":
                s["page"] = "deleted"
            else:
                s["toast"] = "Not permitted"
        return s


class Shop(Family):
    name = "shop"
    GOAL = "Buy the lamp at the lowest total."
    LABELS = {
        "product_link": ["Desk lamp", "Lamp (desk)", "Table lamp"],
        "search_link": ["Search", "Find products", "Browse"],
        "cart_link": ["Cart", "Basket", "Shopping bag"],
        "home_link": ["Home", "Store front", "Main page"],
        "price": ["Price", "Cost", "Amount"],
        "add_btn": ["Add to cart", "Add to basket", "Buy later"],
        "items": ["Items", "Quantity", "Products in cart"],
        "total": ["Total", "Order total", "Sum"],
        "coupon_btn": ["Apply coupon", "Use voucher", "Redeem code"],
        "checkout_btn": ["Checkout", "Place order", "Pay now"],
        "status": ["Status", "Order status", "Result"],
    }
    TITLES = {
        "home": ["Store", "Shop", "Welcome"], "search": ["Search results", "Results", "Matches"],
        "product": ["Product", "Item details", "Details"], "cart": ["Cart", "Basket", "Bag"],
        "done": ["Order placed", "Thank you", "Confirmation"],
    }
    HIDDEN = {"member": [False, True], "in_stock": [True, False]}
    COMMITTAL = {"checkout_btn", "coupon_btn", "add_btn"}

    def init(self, hidden):
        return {"page": "home", "toast": None, "cart": 0, "coupon": False, **hidden}

    def _price(self, s):
        return 8 if s["member"] else 10

    def _total(self, s):
        t = s["cart"] * self._price(s)
        return round(t * (0.8 if s["coupon"] else 1.0), 2)

    def visible(self, s):
        p = s["page"]
        if p == "home":
            return [("product_link", "link", None), ("search_link", "link", None), ("cart_link", "link", None)]
        if p == "search":
            return [("product_link", "link", None), ("home_link", "link", None)]
        if p == "product":
            return [("price", "field", str(self._price(s))), ("add_btn", "button", None),
                    ("home_link", "link", None)]
        if p == "cart":
            return [("items", "field", str(s["cart"])), ("total", "field", str(self._total(s))),
                    ("coupon_btn", "button", None), ("checkout_btn", "button", None),
                    ("home_link", "link", None)]
        if p == "done":
            return [("status", "field", "paid " + str(self._total(s)))]
        raise ValueError(p)

    def transition(self, s, e):
        if e == "product_link":
            s["page"] = "product"
        elif e == "search_link":
            s["page"] = "search"
        elif e == "cart_link":
            s["page"] = "cart"
        elif e == "home_link":
            s["page"] = "home"
        elif e == "add_btn":
            if s["cart"] < 2:
                s["cart"] += 1
                s["toast"] = "Added"
            else:
                s["toast"] = "Limit reached"
        elif e == "coupon_btn":
            if s["cart"] > 0 and not s["coupon"]:
                s["coupon"] = True
                s["toast"] = "Coupon applied"
            else:
                s["toast"] = "Not applicable"
        elif e == "checkout_btn":
            if s["cart"] == 0:
                s["toast"] = "Cart empty"
            elif not s["in_stock"]:
                s["toast"] = "Out of stock"
            else:
                s["page"] = "done"
        return s


class Mail(Family):
    name = "mail"
    GOAL = "Send the report to Dana."
    LABELS = {
        "compose_link": ["Compose", "New message", "Write"],
        "outbox_link": ["Outbox", "Sent items", "Outgoing"],
        "inbox_link": ["Inbox", "Back to inbox", "Messages"],
        "to_field": ["To", "Recipient", "Send to"],
        "fill_to": ["Pick Dana from contacts", "Add Dana", "Choose recipient: Dana"],
        "send_later_btn": ["Send later", "Schedule send", "Delay send"],
        "send_now_btn": ["Send now", "Send", "Deliver"],
        "scheduled": ["Scheduled", "Queued", "Pending"],
        "sent": ["Sent", "Delivered", "Done"],
        "cancel_btn": ["Cancel scheduled", "Unschedule", "Stop sending"],
    }
    TITLES = {"inbox": ["Inbox", "Mail", "Messages"], "compose": ["Compose", "New mail", "Draft"],
              "outbox": ["Outbox", "Outgoing", "Send queue"]}
    HIDDEN = {"offline": [False, True]}
    COMMITTAL = {"send_later_btn", "send_now_btn", "cancel_btn"}
    DELAY = 3

    def init(self, hidden):
        return {"page": "inbox", "toast": None, "to": "", "countdown": 0, "sent": 0, **hidden}

    def visible(self, s):
        p = s["page"]
        if p == "inbox":
            return [("compose_link", "link", None), ("outbox_link", "link", None)]
        if p == "compose":
            return [("to_field", "field", s["to"] or "-"), ("fill_to", "button", None),
                    ("send_later_btn", "button", None), ("send_now_btn", "button", None),
                    ("inbox_link", "link", None)]
        if p == "outbox":
            v = [("scheduled", "field", str(int(s["countdown"] > 0))), ("sent", "field", str(s["sent"]))]
            if s["countdown"] > 0:
                v.append(("cancel_btn", "button", None))
            v.append(("inbox_link", "link", None))
            return v
        raise ValueError(p)

    def transition(self, s, e):
        if e == "compose_link":
            s["page"] = "compose"
        elif e == "outbox_link":
            s["page"] = "outbox"
        elif e == "inbox_link":
            s["page"] = "inbox"
        elif e == "fill_to":
            s["to"] = "Dana"
        elif e == "send_later_btn":
            if not s["to"]:
                s["toast"] = "Add a recipient"
            elif s["countdown"] > 0:
                s["toast"] = "Already scheduled"
            else:
                s["countdown"] = self.DELAY + 1  # tick() below decrements in the same step
                s["toast"] = "Scheduled"
        elif e == "send_now_btn":
            if not s["to"]:
                s["toast"] = "Add a recipient"
            elif s["offline"]:
                s["toast"] = "Send failed"
            else:
                s["sent"] += 1
                s["toast"] = "Sent"
        elif e == "cancel_btn":
            s["countdown"] = 0
            s["toast"] = "Cancelled"
        return s

    def tick(self, s):
        if s["countdown"] > 0:
            s["countdown"] -= 1
            if s["countdown"] == 0 and not s["offline"]:
                s["sent"] += 1
        return s


class Ticket(Family):
    name = "ticket"
    GOAL = "Get the access request approved."
    LABELS = {
        "new_link": ["New request", "Request access", "Create ticket"],
        "status_link": ["My requests", "Request status", "Tickets"],
        "profile_link": ["Profile", "About me", "My details"],
        "dash_link": ["Dashboard", "Back", "Overview"],
        "submit_btn": ["Submit", "Send request", "File ticket"],
        "req_status": ["Request", "Ticket state", "Approval"],
        "escalate_btn": ["Escalate", "Ask manager", "Expedite"],
        "perm": ["Escalation rights", "Can escalate", "Manager access"],
    }
    TITLES = {"dash": ["Dashboard", "Overview", "Home"], "new": ["New request", "Create", "Request form"],
              "status": ["Requests", "Status", "Tickets"], "profile": ["Profile", "Me", "Account"]}
    HIDDEN = {"approve_delay": [2, 4], "can_escalate": [False, True]}
    COMMITTAL = {"submit_btn", "escalate_btn"}

    def init(self, hidden):
        return {"page": "dash", "toast": None, "req": "none", "countdown": 0, **hidden}

    def visible(self, s):
        p = s["page"]
        if p == "dash":
            return [("new_link", "link", None), ("status_link", "link", None), ("profile_link", "link", None)]
        if p == "new":
            return [("submit_btn", "button", None), ("dash_link", "link", None)]
        if p == "status":
            v = [("req_status", "field", s["req"])]
            if s["req"] == "pending":
                v.append(("escalate_btn", "button", None))
            v.append(("dash_link", "link", None))
            return v
        if p == "profile":
            return [("perm", "field", "yes" if s["can_escalate"] else "no"), ("dash_link", "link", None)]
        raise ValueError(p)

    def transition(self, s, e):
        if e == "new_link":
            s["page"] = "new"
        elif e == "status_link":
            s["page"] = "status"
        elif e == "profile_link":
            s["page"] = "profile"
        elif e == "dash_link":
            s["page"] = "dash"
        elif e == "submit_btn":
            if s["req"] == "none":
                s["req"] = "pending"
                s["countdown"] = s["approve_delay"] + 1
                s["toast"] = "Submitted"
            else:
                s["toast"] = "Already submitted"
        elif e == "escalate_btn":
            if s["can_escalate"]:
                s["req"] = "approved"
                s["countdown"] = 0
                s["toast"] = "Escalated"
            else:
                s["toast"] = "Denied"
        return s

    def tick(self, s):
        if s["countdown"] > 0:
            s["countdown"] -= 1
            if s["countdown"] == 0 and s["req"] == "pending":
                s["req"] = "approved"
        return s


FAMILIES = {f.name: f for f in (Settings(), Shop(), Mail(), Ticket())}


# ----------------------------------------------------------------------------------------------
# Core semantics
# ----------------------------------------------------------------------------------------------


def sem_obs(fam: Family, s) -> tuple:
    return (s["page"], tuple(fam.visible(s)), s["toast"])


def actions(fam: Family, s) -> list:
    return [e for e, role, _ in fam.visible(s) if role in CLICK_ROLES]


def step(fam: Family, s, eid):
    if eid not in actions(fam, s):
        raise ValueError(f"action {eid} not available on page {s['page']}")
    s = dict(s)
    s["toast"] = None
    s = fam.transition(s, eid)
    return fam.tick(s)


def state_key(s) -> str:
    return h(s)


def run(fam: Family, hidden: dict, seq: list):
    """Execute a semantic action sequence; return list of states s_0..s_T."""
    s = fam.init(dict(hidden))
    states = [s]
    for a in seq:
        s = step(fam, s, a)
        states.append(s)
    return states


def h_equivalent(fam: Family, s1, s2, H: int, nocommit: bool = False):
    """H-step trace equivalence. Returns (bool, witness action sequence or None).
    If nocommit, only non-committal actions are explored (a weaker, probe-only notion)."""
    if sem_obs(fam, s1) != sem_obs(fam, s2):
        return False, []
    if H == 0:
        return True, None
    a1, a2 = actions(fam, s1), actions(fam, s2)
    if a1 != a2:
        return False, []
    for a in a1:
        if nocommit and a in fam.COMMITTAL:
            continue
        ok, w = h_equivalent(fam, step(fam, s1, a), step(fam, s2, a), H - 1, nocommit)
        if not ok:
            return False, [a] + w
    return True, None


# ----------------------------------------------------------------------------------------------
# Rendering (what the agent sees)
# ----------------------------------------------------------------------------------------------

DECOR_TRAIN = ["Spring sale: 20% off selected items", "Last synced 10:42", "Tip: press ? for shortcuts",
               "You have 3 unread notifications"]
DECOR_TEST = ["Maintenance window Sunday 02:00", "New: dark mode is here", "Server time 17:05 UTC"]


@dataclass
class RenderCfg:
    layout_seed: int | None = None   # None = canonical element order
    label_var: int = 0               # 0 = canonical labels; 1/2 = synonyms
    decor: list = field(default_factory=list)
    style: bool = False

    def as_dict(self):
        return {"layout_seed": self.layout_seed, "label_var": self.label_var, "decor": list(self.decor),
                "style": self.style}


STYLE = {"button": " {class=btn btn-primary}", "link": " {class=nav-link}", "checkbox": " {class=form-check}",
         "field": " {class=text-muted}"}


def label(fam, eid, cfg):
    return fam.LABELS[eid][cfg.label_var]


def render(fam: Family, so: tuple, cfg: RenderCfg, t: int = 0) -> str:
    page, vis, toast = so
    lines = []
    for eid, role, val in vis:
        ln = f"[{role}] {label(fam, eid, cfg)}"
        if val is not None:
            ln += f": {val}"
        if cfg.style:
            ln += STYLE.get(role, "")
        lines.append(ln)
    if cfg.layout_seed is not None:
        random.Random(cfg.layout_seed * 7919 + int(h(page), 16) % 997).shuffle(lines)  # stable hash
    out = [f"# {fam.TITLES[page][cfg.label_var]}"]
    if toast:
        out.append(f"[status] {toast}")
    for i, d in enumerate(cfg.decor):
        lines.insert(min(len(lines), 1 + i), f"[static] {d}")
    return "\n".join(out + lines)


def render_history(fam: Family, hidden: dict, seq: list, cfg: RenderCfg):
    """Return (obs_texts[0..T], action_texts[0..T-1], final_state)."""
    states = run(fam, hidden, seq)
    obs = [render(fam, sem_obs(fam, s), cfg) for s in states]
    acts = [f"click({label(fam, a, cfg)})" for a in seq]
    return obs, acts, states[-1]
