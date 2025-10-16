// Loads the JSON and gives you precomputed structures for both the jsMind tree and the table.
import { useEffect, useMemo, useState } from 'react';
import { DEFAULT_TOPIC } from '../utils/type';
import type { SchemaDoc } from '../utils/type';

export function useSchema() {
  const [schema, setSchema] = useState<SchemaDoc | null>(null);
  const [status, setStatus] = useState<'idle'|'loading'|'error'|'loaded'>('idle');

  useEffect(() => {
    let cancelled = false;
    setStatus('loading');
    fetch('/taxonomy/schema.json', { cache: 'no-store' })
      .then(r => { if (!r.ok) throw new Error(String(r.status)); return r.json(); })
      .then((json: SchemaDoc) => { if (!cancelled){ setSchema(json); setStatus('loaded'); }})
      .catch(() => { if (!cancelled){ setStatus('error'); }});
    return () => { cancelled = true; };
  }, []);

  const mind = useMemo(() => {
    console.log('Generating mind from schema', schema);
    const taxonomy = Array.isArray(schema?.Taxonomy) ? schema!.Taxonomy! : [];
    return {
      meta: { name: 'Taxon Mind Map', author: 'TaxonAI', version: '1.0' },
      format: 'node_tree',
      data: {
        id: 'root',
        topic: DEFAULT_TOPIC,
        children: taxonomy.map((aspect, i) => ({
          id: `aspect-${i}`,
          topic: aspect.Aspect || `Aspect ${i + 1}`,
          description: aspect.Description || '',
          direction: i % 2 === 0 ? 'left' : 'right',
          children: (aspect.Options ?? []).map((opt, j) => ({ id: `aspect-${i}-opt-${j}`, topic: opt }))
        }))
      }
    } as const;
  }, [schema]);

  return { schema, mind, status };
}