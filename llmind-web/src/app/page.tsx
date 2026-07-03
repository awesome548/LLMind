import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/src/components/ui/card';

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center px-4 py-12">
      <section className="mb-8 space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">LLMind</h1>
        <p className="text-muted-foreground">
          A research prototype for LLM-assisted design-space exploration in media
          architecture — a living design-space schema with an evidence map,
          precedent-grounded generation, and design candidates.
        </p>
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Open the workspace</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            Structure (tree · schema · cross-tab), the Design Space map, and
            Perspectives — one shared exploration state.
          </p>
          <Link
            href="/mindmap"
            className="text-sm font-semibold underline underline-offset-4"
          >
            Enter LLMind →
          </Link>
        </CardContent>
      </Card>
    </main>
  );
}
