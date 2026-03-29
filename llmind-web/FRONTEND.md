frontend/
├── src/
│   ├── app/                # Next.js App Router (File-based Routing)
│   │   ├── (auth)/         # Grouped routes (Login, Signup)
│   │   ├── dashboard/      # /dashboard page
│   │   └── layout.tsx      # Global providers (TanStack, etc.)
│   ├── components/
│   │   ├── ui/             # shadcn/ui (Atomic components)
│   │   └── shared/         # Reusable patterns (Header, Footer)
│   ├── features/           # BUSINESS LOGIC (The core of your MVP)
│   │   ├── user-profile/
│   │   │   ├── components/ # Feature-specific UI
│   │   │   ├── hooks/      # TanStack Query logic
│   │   │   └── services/   # API calls to Python backend
│   ├── store/              # ZUSTAND STORES
│   │   ├── useUserStore.ts
│   │   └── useUIStore.ts
│   ├── types/              # Auto-generated from Python OpenAPI
│   └── lib/                # Utils (api-client.ts, utils.ts)
├── bun.lockb
└── package.json

Category,Recommended Tool,2026 Advantage
Runtime,Bun,"Instant startup, native TS, fastest installs."
Framework,Next.js 16,"Stable RSC, React Compiler (No more useMemo)."
Logic Layer,TanStack Query,High-performance server-state & cache sync.
Global Store,Zustand,"Minimalist, high-speed client-side state."
API Bridge,openapi-typescript,Eliminates manual type-definitions.
Styling,Tailwind CSS,Best-in-class AI-code generation compatibility.

### OpenAPI type generation
bunx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts