import Link from 'next/link';
import { Card, CardContent, CardHeader, CardTitle } from '@/src/components/ui/card';

const links = [
  {
    href: '/mindmap',
    title: 'Mind Map Demo',
    description: 'Simple topic selection with a related project panel.',
  },
  {
    href: '/projects',
    title: 'Projects Demo',
    description: 'Simple project panel page with sample data.',
  },
] as const;

export default function HomePage() {
  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 py-12">
      <section className="mb-8 space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">LLMind Web</h1>
        <p className="text-muted-foreground">
          Minimal pages based on the MindMap and ProjectPanel concepts.
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {links.map((link) => (
          <Card key={link.href}>
            <CardHeader>
              <CardTitle>{link.title}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground">{link.description}</p>
              <Link href={link.href} className="text-sm font-semibold underline underline-offset-4">
                Open page
              </Link>
            </CardContent>
          </Card>
        ))}
      </section>
    </main>
  );
}
