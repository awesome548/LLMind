import { useCallback, useMemo, useState } from 'react';
import { TabBar } from './components/TabBar';
import { MindMap } from './components/MindMap';
import { SchemaTable } from './components/SchemaTable';
import { ProjectPanel } from './components/ProjectPanel';
import { DEFAULT_TOPIC } from './utils/type';
import { useSchema } from './hooks/useSchema';
import { useProjects } from './hooks/useProjects';

export default function App() {
  const [activeTab, setActiveTab] = useState<'mindmap'|'table'>('mindmap');
  const [contextText, setContextText] = useState<string>(DEFAULT_TOPIC);
  const [contextDescription, setContextDescription] = useState<string>('');
  const { schema, mind, status } = useSchema();
  const { projects, search, statusText, isLoading } = useProjects();

  const normalize = (s: string) => s?.normalize('NFKC').replace(/\s+/g, ' ').trim().toLowerCase();
  const handleSelect = useCallback((topic: string, lineage: string[], userInitiated: boolean) => {
    const hierarchySegments = (Array.isArray(lineage) ? lineage : []).filter(Boolean);
    const label = hierarchySegments.length ? hierarchySegments.slice(1).join(' > ') : (topic || DEFAULT_TOPIC);
    setContextText(label);

    // If schema hasn't loaded yet, clear and bail early
    if (!schema?.Taxonomy?.length) {
      setContextDescription('');
      search(topic, lineage, userInitiated);
      return;
    }

    console.log(schema);

    const aspects = schema.Taxonomy;
    const aspectNames = new Set(aspects.map(a => normalize(a.Aspect)));

    // Try to find any lineage segment that is a known Aspect (don’t assume position)
    const lineageAspect =
      hierarchySegments.find(seg => aspectNames.has(normalize(seg))) ||
      (aspectNames.has(normalize(topic)) ? topic : undefined);
    
    console.log({ lineage, lineageAspect })

    let matchingAspect = undefined;
    if (lineageAspect) {
      const target = normalize(lineageAspect);
      matchingAspect = aspects.find(a => normalize(a.Aspect) === target);
    }

    setContextDescription(matchingAspect?.Description ?? '');

    search(topic, lineage, userInitiated);
  }, [schema, search, setContextText, setContextDescription]);

  const statusMessage = useMemo(() => {
    if (status === 'loading') return 'Loading schema…';
    if (status === 'error') return 'Unable to load schema';
    return 'Schema loaded';
  }, [status]);

  return (
    <div className="page-shell">
      <div className="page-layout">
        <main className="mindmap-wrapper">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="tab-panels">
            <MindMap active={activeTab === 'mindmap'} mind={mind} onSelect={handleSelect} statusText={statusMessage} />
            <SchemaTable active={activeTab === 'table'} schema={schema} statusText={statusMessage} />
          </div>
        </main>
        <ProjectPanel
          projects={projects}
          contextText={contextText}
          contextDescription={contextDescription}
          statusText={statusText}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}
