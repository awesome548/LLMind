sample zustand store syntax
```
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface User {
    id: string;
    name: string;
    email: string;
}

interface UserState {
    user: User | null;
    isAuthenticated: boolean;
    setUser: (user: User | null) => void;
}

export const useUserStore = create<UserState>()(
    devtools(
        persist(
            (set) => ({
                user: null,
                isAuthenticated: false,
                setUser: (user) => set({ user, isAuthenticated: !!user }),
            }),
            { name: 'user-storage' } // Persists to LocalStorage automatically
        )
    )
);
```