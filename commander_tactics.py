TACTIC_LIBRARY = {
    "COVER_AND_FIRE": {
        "summary": "Use hard cover to break enemy sight lines, peek to fire, then return behind cover.",
        "roles": {
            "COVER_SHOOTER": "Take hard cover, peek from a corner, fire, and recover behind cover.",
            "COVER_SUPPORT": "Occupy separate cover and alternate fire with the assault element.",
            "SECURITY": "Watch an uncovered approach while remaining near protected positions.",
        },
        "required_roles": {"COVER_SHOOTER"},
    },
    "DIVERSION_AND_FLANK": {
        "summary": "Expose a controlled decoy element to draw enemy attention while infiltrators maneuver through a low-exposure route to the flank.",
        "roles": {
            "DECOY": "Fight from cover and remain intermittently visible so Red aims at this element; withdraw only at danger-close range or after taking damage.",
            "INFILTRATOR": "Use the lowest-exposure obstacle route to reach the enemy flank before engaging at close range.",
            "OVERWATCH": "Occupy separate cover and fire on enemies that threaten the decoy or infiltrators.",
        },
        "required_roles": {"DECOY", "INFILTRATOR"},
    },
    "FIX_AND_FLANK": {
        "summary": "Fix the enemy with part of the force while flankers maneuver around it.",
        "roles": {
            "FIXER": "Maintain contact and pin the enemy.",
            "FLANKER": "Maneuver around the enemy before engaging.",
            "SUPPORT": "Support the fixing or flanking element as needed.",
        },
        "required_roles": {"FIXER", "FLANKER"},
    },
    "FOCUS_FIRE": {
        "summary": "Concentrate available fire on a common target until it is destroyed.",
        "roles": {
            "SHOOTER": "Engage the common priority target.",
            "SECURITY": "Protect the force while preserving the firing position.",
        },
        "required_roles": {"SHOOTER"},
    },
    "SEARCH_AND_BACK": {
        "summary": "Scout for Red forces and return to rejoin the reserve after contact.",
        "roles": {
            "SCOUT": "Search for Red, then return toward the reserve after contact.",
            "RESERVE": "Remain grouped as the scout's return element.",
            "SECURITY": "Protect the reserve and returning scout.",
        },
        "required_roles": {"SCOUT", "RESERVE"},
    },
    "REGROUP": {
        "summary": "Concentrate dispersed Blue units around an anchor before further action.",
        "roles": {
            "ANCHOR": "Hold as the rally point for the force.",
            "REGROUPER": "Move toward the anchor and rejoin the force.",
            "SECURITY": "Cover the regrouping force.",
        },
        "required_roles": {"ANCHOR"},
    },
}


def tactic_library_prompt() -> str:
    """전술 이름과 역할 설명을 짧은 프롬프트 문자열로 만든다."""
    lines = []
    for tactic, definition in TACTIC_LIBRARY.items():
        roles = ", ".join(
            f"{role}: {description}"
            for role, description in definition["roles"].items()
        )
        lines.append(f"- {tactic}: {definition['summary']} Roles=[{roles}]")
    return "\n".join(lines)


def selected_tactic_prompt(tactic: str) -> str:
    """선택된 전술 하나의 목적과 역할 원칙을 프롬프트로 만든다."""
    definition = TACTIC_LIBRARY[tactic]
    roles = "\n".join(
        f"- {role}: {description}"
        for role, description in definition["roles"].items()
    )
    return f"{tactic}: {definition['summary']}\n{roles}"
