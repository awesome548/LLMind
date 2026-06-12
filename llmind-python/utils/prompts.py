SYSTEM_PROMPT = {
  "project": "The project aims to create a large-scale interactive media installation that engages the public and contributes to the cultural program of Aarhus 2017. The installation will activate the city's greenhouse dome, transforming it into a living canvas of light that symbolizes growth, cycles, and participation. At its core, the project seeks to enable citizens to interact with the greenhouse facade in ways that foster a sense of participation and ownership, while also exploring the intersection of interaction design, light art, and architecture through iterative prototyping and experimentation. In doing so, it reflects themes of sustainability, growth, and community in direct alignment with the ambitions of the European Capital of Culture program.",
  "system": "You are a creative professional designer and analytical thinker. You use concise expression with maximum information density that are highly based on facts and data."
}

IDEA_FIRST_PROMPT = """Here are the existing artefacts:
'''
{existing_artefacts}
'''
TASK:
Based on the existing artefacts, captures the salient components of its design space.
Ideate creative and concrete Aspects / Options to understand the design space.

DEFINITIONS:
- Design Space: a conceptual space, which encompasses the creativity constraints that govern what the outcome of the design process might (and might not) be.

Respond in the following format:
THOUGHT:
<Your reasoning: high-level design plan. Explain how your constraints sparks creativity, and what outcomes you aim to enable.>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new ideas in JSON format with the following fields under the root "Taxonomy":
- Aspect: a key dimension or parameter of the design space (e.g., display technology, location, type of interaction).
- Option: possible alternatives for an aspect (e.g., for the aspect “Display,” options might include LED panels or projection).
For each Aspect, provide comprehensive Options. 
Nor futher nesting is allowed.

This JSON will be automatically parsed, so ensure the format is precise."""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, Consider:
- Aspects and options define the range of possible paths through the design space.
- They serve both as enablers of creative thinking (by outlining possibilities) and constraints (by ruling out alternatives when one option is chosen) 

TASK:
Refine and consolidate them into more meaningful ones while keeping the spirit of the original idea.
Both Aspects and Options should be concise and clear.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
In <JSON>, provide the refined ideas in JSON format that strictly follows the same structure as before.:

Ensure the idea is clear and the JSON format is correct."""


USER_PROMPT_TEMPLATE = """TASK:
Explore into the given Aspect in the current Taxonomy, suggest new simple Options that:
- define the range of possible paths through the design space.
- serve both as enablers of creative thinking (by outlining possibilities) and constraints (by ruling out alternatives when one option is chosen)

ASPECT TO EXPLORE:
ID: {{SELECTED_NODE_ID}}
Topic: {{SELECTED_NODE_TOPIC}}

RELATED PROJECTS:
{{RELATED_PROJECTS}}

TAXONOMY:
{{TAXONOMY}}

Respond in the following format
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
In <JSON> provide ID of the node that you explored and create new Node ids accordingly, and Options.
Suppose explored node is "Aspect", each of new Nodes is "Option":
- Aspect: a key dimension or parameter of the design space (e.g., display technology, location, type of interaction).
- Option: possible alternatives for an aspect (e.g., for the aspect "Display," options might include LED panels or projection).
the options should be concise and simple; each "desc" is 2-4 sentences written like a real project description (a catalog entry of a built installation): name the mechanism, material or technology, the interaction, and the audience experience or context. It is embedded to position the idea among real project descriptions — concrete nouns, no marketing fluff.

Use the following JSON structure exactly:
{
  "parent_id": "<SELECTED_NODE_ID>",
  "options": [
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<project-style description>" },
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<project-style description>" }
  ]
}"""


# Drafts a candidate's BRIEF (its identity layer) from its committed choices —
# the starting point the designer edits, never the final word (Part 10 I1).
# Output is plain text (no JSON): the brief is embedded as-is for placement, so
# it must read like the corpus's own project descriptions.
DRAFT_BRIEF_PROMPT = """TASK:
Write the project description (brief) for ONE new media architecture design that
commits to all of the choices below.

PROJECT CONTEXT (the designer's overall project; may be empty):
{{PROJECT_OVERVIEW}}

COMMITTED CHOICES (one per design dimension):
{{CHOICES}}

INSTRUCTIONS:
- 3-5 sentences, written like a real project description (the catalog entry of a
  built installation): name the mechanism, material or technology, the
  interaction, and the audience experience or context. Concrete nouns, no
  marketing fluff.
- Every committed choice must be EMBODIED in the design — woven into one
  coherent concept, not listed.
- Do not add commitments that contradict the choices.
- Respond with ONLY the description text: no headings, no quotes, no preamble.
"""


# Part 12 A2: corpus annotation — one call per OPTION over its embedding
# shortlist. Membership judgment ("genuinely exemplifies"), not similarity:
# the shortlist already handled aboutness; the LLM decides exemplification.
ANNOTATE_OPTION_PROMPT = """TASK:
You are annotating a corpus of real media-architecture projects against ONE
design option. Decide, for each project, whether the project GENUINELY
EXEMPLIFIES the option — it must actually have/do this, not merely relate to
the theme.

DESIGN OPTION:
{{OPTION_NAME}}: {{OPTION_DESC}}

PROJECTS:
{{PROJECTS}}

INSTRUCTIONS:
- Judge from BOTH the concept text and the [Details: ...] technical notes —
  the technical notes often name the display/interaction technology.
- Be strict about the option's substance, not its exact wording: a project
  exemplifies the option when its text clearly indicates it.
- Respond with ONLY a JSON array of the numbers of the exemplifying projects,
  e.g. [2, 5, 11]. An empty array [] is a valid answer. No other text.
"""


# Part 12 B2: cross-tab cell generation — one call per EMPTY option×option
# cell. Morphological combination (Halskov: empty cells are exact, nameable
# gaps): the prompt names the gap and seeds with half-matching precedents.
GENERATE_CELL_PROMPT = """TASK:
You are exploring a design space of media architecture. In the corpus of real
projects, NO precedent combines these two commitments — this is an exact,
nameable gap:

- {{ASPECT_A}}: {{OPTION_A_NAME}} — {{OPTION_A_DESC}}
- {{ASPECT_B}}: {{OPTION_B_NAME}} — {{OPTION_B_DESC}}

NEARBY PRECEDENTS (each satisfies ONE of the two, not both):
{{EXEMPLARS}}

INSTRUCTIONS:
- Propose exactly ONE new project concept that genuinely commits to BOTH.
- Write it in the same register as the precedents: a concrete site, medium,
  and behaviour — not a theme statement.
- desc is 2-4 sentences, in the style of a real project description.
- Respond with ONLY a JSON object: {"name": "...", "desc": "..."}. No other
  text.
"""


# Part 12 B3: steering — ONE deliberate move on a candidate's brief, made in
# language (the evidence rule: deltas brief the LLM and measure the result;
# embeddings never construct text).
STEER_PROMPT = """TASK:
Revise a media-architecture design brief by ONE deliberate move, keeping its
identity otherwise intact.

CURRENT BRIEF:
{{BRIEF}}

THE MOVE:
{{MOVE}}

PRESERVE (do not weaken these):
{{PRESERVE}}

INSTRUCTIONS:
- Same project, same register, similar length.
- Make the move CONCRETE: change materials, behaviours, siting, or program —
  not adjectives.
- Respond with ONLY a JSON object:
  {"revised_brief": "...", "named_qualities": ["...", "..."]}
  where named_qualities are 1-3 short names for the qualities the move added
  or strengthened.
"""


# Part 12 C2: burden-inverted reflection capture — the system drafts the
# one-line rationale a designer might write for what they just did; the
# designer accepts, edits, or skips it. Never authoritative, always editable.
REFLECT_PROMPT = """TASK:
A designer exploring a media-architecture design space just did this:

{{EVENT}}

Draft the ONE-LINE rationale they might note down for it.

INSTRUCTIONS:
- First person, specific to this act, under 18 words.
- A reason or intention, not a description of the act itself.
- Respond with ONLY the sentence. No preamble, no quotes.
"""


# Version tag logged with every generate-at call so prompt/seeding variants can
# be compared in data/projection/generate_log.jsonl. Bump when the prompt or
# the seeding strategy changes behaviour.
# v3: descs became 2-4 sentence project-style text (register-gap fix, Part 9 H1).
# v4: optional DESIGNER_BRIEF context block (squiggle hypothesis, Part 10 I2).
GENERATE_AT_PROMPT_VERSION = 4

GENERATE_AT_PROMPT = """TASK:
The designer clicked an UNOCCUPIED location on a 2D map of the design space — a
gap where no real project and no existing idea sits. Propose new design Options
that conceptually belong AT that location.

SURROUNDING REAL PROJECTS (the gap's neighbourhood — these bracket the clicked
location from different sides):
{{RELATED_PROJECTS}}

NEARBY EXISTING IDEAS (already on the map near the click — do NOT duplicate or
trivially rephrase any of these):
{{NEARBY_OPTIONS}}

DESIGNER'S CURRENT CONCEPT (their work-in-progress design — context only: use it
to keep options relevant to their project, but do NOT restate, refine, or stay
close to it; the task is still to fill the map gap):
{{DESIGNER_BRIEF}}

PARENT ASPECT (the new Options will be filed under this dimension):
ID: {{SELECTED_NODE_ID}}
Topic: {{SELECTED_NODE_TOPIC}}

TAXONOMY (context only):
{{TAXONOMY}}

INSTRUCTIONS:
- Propose Options that sit conceptually BETWEEN the surrounding projects yet are
  distinct from every one of them: fill the gap, do not imitate the nearest
  project and do not average them into something generic.
- Each Option must be a plausible alternative within the parent Aspect.
- Each "desc" is 2-4 sentences written like a real project description (a
  catalog entry of a built installation): name the mechanism, material or
  technology, the interaction, and the audience experience or context. It is
  embedded to position the idea on the map among real project descriptions —
  concrete nouns, no marketing fluff.
- 2 to 4 Options; concise topics (2-5 words).

Respond in the following format
THOUGHT:
<Why these options fill the gap between the surrounding projects.>

NEW IDEA JSON:
```json
<JSON>
```
Use the following JSON structure exactly:
{
  "parent_id": "<SELECTED_NODE_ID>",
  "options": [
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<project-style description>" }
  ]
}"""