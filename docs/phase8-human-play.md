# Phase 8 — Playing humans (prepared, NOT executed)

Everything in this file is ready to go and deliberately **not done**.
Phase 8 is the first phase whose actions leave this machine: it contacts
a real person and creates an account on someone else's service. Those are
the user's calls, not the agent's, so this documents the plan and drafts
the message rather than sending anything.

## What is blocked on a human decision

1. **Emailing Juho Snellman** (terra.snellman.net's operator) to ask
   about bot access. Draft below — review, edit, and send it yourself.
2. **Creating any account** on terra.snellman.net.
3. **Playing rated games against real people.** Even with permission,
   entering a bot into games that affect other players' ratings is a
   social act with consequences for them, not just for us.

## Why asking first is the right move (not just politeness)

The crawler already respects the site's boundaries deliberately (the
`/app/` endpoints are robots-disallowed, so the crawl runs at <=1 req/s
with an identifying User-Agent and a resumable cache). Turning up
unannounced with a bot account after that would undo the good standing
that made the corpus possible. The operator also has information we
don't: whether bot games skew the rating pool, whether there is an
existing sandbox, and whether a specific game mode is acceptable.

## Draft message (for the user to send, edit freely)

> Subject: Asking permission before running an AI player on terra.snellman.net
>
> Hi Juho,
>
> I'm a student working on a Terra Mystica AI as a personal research
> project. I've built a rules engine validated against the per-move
> ledger deltas from tournament games on your site (3,552 of 3,553
> games replay with zero mismatches — your ledger format made this
> possible, thank you), and trained a policy on Div 1–3 tmtour games.
>
> I'd like to eventually see how it does against real players, and I
> don't want to do anything on your site that you haven't agreed to.
> Before I go any further:
>
> 1. Is there any acceptable way to run a bot account, and if so under
>    what constraints (unrated games only, a marked account name, a
>    request-rate limit, specific game modes)?
> 2. Would you prefer I not do this at all? That's a completely fine
>    answer and I'll respect it.
> 3. Separately: my crawler fetches game logs at <=1 request/second with
>    an identifying User-Agent, caching so it never refetches. If that
>    is still too aggressive, tell me and I'll slow it down or stop.
>
> Happy to share results or the engine's validation numbers if useful.
>
> Thanks for keeping the site and its data available,
> Keerthik

## If permission is granted — what is already built vs still needed

**Already built:**
- The agent interface, so any agent (imitation, MCTS, LLM) can play.
- A full rules engine to validate and construct legal moves.
- Reproducible seeding, so any game we play can be replayed exactly.

**Still needed:**
1. **A move-proxy adapter** — the agent proposes, a human submits, so no
   automated posting to the site is required. This is the lowest-risk
   first step and needs no bot account at all.
2. **Draft-mode arena** (decision D4.3): real games start with a faction
   draft the agent currently cannot do. Against humans the draft is
   played, so it has to be modeled or delegated to the operator.
3. **A rating-comparison design.** Snellman ratings are faction-adjusted
   Elo; our arena reports mean placement and TrueSkill. Comparing them
   needs a stated mapping, ideally set before playing rather than after
   seeing results.

## Fallback if the answer is no

Self-hosted games with volunteer opponents, or manual move proxying in
casual games with consenting players. Neither requires the operator's
infrastructure, and both preserve the "vs. real humans" measurement the
master plan wants. The corpus itself remains untouched by this decision:
it was gathered under the site's published rules and needs no further
access.
