# llmind-web — LLMind frontend

The browser client of **LLMind**, a research prototype for LLM-assisted
design-space exploration (media-architecture domain). Next.js 16 · React 19 ·
Bun · Zustand · TanStack Query.

> This file is only a launcher. The real documentation:
>
> | Doc | Contents |
> |---|---|
> | [`FRONTEND.md`](FRONTEND.md) | Architecture, component map, feature flows, the locked design language |
> | [`ZUSTAND.md`](ZUSTAND.md) | Store shape, actions, persistence (the whole exploration state) |
> | [`REACT-QUERY.md`](REACT-QUERY.md) | Every query/mutation hook and the endpoint it wraps |
> | [`../PROJECT-REPORT.md`](../PROJECT-REPORT.md) | What the system is, feature-by-feature, with the research argument |

## Run

```bash
bun install
bun dev          # → http://localhost:3000  (main page: /mindmap)
```

The frontend calls the FastAPI backend **directly** at
`NEXT_PUBLIC_API_BASE_URL ?? http://localhost:8000` (not through the Next.js
rewrite proxy — long local-LLM responses don't survive it; see the root
`CLAUDE.md`). Start the backend first (`../llmind-python`), and the local
model server (LM Studio) if you want generation/annotation features.

## Test / lint

```bash
bun test src     # 66 unit tests (pure utils + store invariants)
bun lint
```
