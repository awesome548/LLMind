'use client';

// App-Router render error boundary. A render throw (e.g. a malformed slice that
// slips past the session sanitizer, or any future bug) would otherwise blank the
// screen mid-session. Because the store persists to localStorage, "Reload view"
// loses nothing — the exploration is recovered on remount (ITERATION-M M-E4).

import { useEffect } from 'react';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Surfaced in the console (and any study screen recording) with a stable tag.
    console.error('[LLMind] render error:', error);
  }, [error]);

  return (
    <main className="flex h-screen w-full flex-col items-center justify-center gap-4 bg-background p-8 text-center">
      <div className="max-w-md space-y-3">
        <h1 className="text-lg font-semibold text-foreground">Something went wrong rendering the view.</h1>
        <p className="text-sm text-muted-foreground">
          Your exploration is saved locally — reloading the view loses nothing.
        </p>
        {error?.message ? (
          <pre className="max-h-32 overflow-auto rounded-lg border bg-muted/50 p-3 text-left text-xs text-muted-foreground">
            {error.message}
          </pre>
        ) : null}
        <button
          type="button"
          onClick={reset}
          className="rounded-lg border bg-background px-4 py-2 text-sm font-semibold text-foreground shadow-sm transition-colors hover:bg-muted"
        >
          Reload view
        </button>
      </div>
    </main>
  );
}
