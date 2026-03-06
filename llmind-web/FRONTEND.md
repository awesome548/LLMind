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