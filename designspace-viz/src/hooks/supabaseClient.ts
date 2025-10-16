import { createClient, type SupabaseClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL as string;
const anonKey = import.meta.env.VITE_SUPABASE_KEY as string;

// Put the client on globalThis so Vite HMR doesn't create duplicates in dev.
const GLOBAL_KEY = '__supabase_singleton__';

type GlobalWithSB = typeof globalThis & {
  [GLOBAL_KEY]?: SupabaseClient;
};

const g = globalThis as GlobalWithSB;

export const supabase: SupabaseClient =
  g[GLOBAL_KEY] ??
  (g[GLOBAL_KEY] = createClient(url, anonKey, {
    auth: {
      // Use your own stable key so you don't clash with other apps on the same origin
      storageKey: 'myapp-auth',
      persistSession: true,
      autoRefreshToken: true,
      detectSessionInUrl: true,
    },
  }));