// App.tsx
import { useEffect, useMemo } from 'react';
import { TabBar } from './components/TabBar';
import { MindMap } from './components/MindMap';
import { SchemaTable } from './components/SchemaTable';
import { ProjectPanel } from './components/ProjectPanel';
import { useAppStore } from './store/appStore';
import { useSchemaStore } from './store/schemaStore';
import { useProjectStore } from './store/projectStore';
import { DEFAULT_TOPIC } from './types/taxonomy';

export default function App() {
  // Select state from store
  const activeTab = useAppStore(state => state.activeTab);
  const schema = useSchemaStore(state => state.schema);
  const schemaStatus = useSchemaStore(state => state.schemaStatus);
  const projects = useProjectStore(state => state.projects);
  const projectsLoading = useProjectStore(state => state.projectsLoading);
  const projectsStatusText = useProjectStore(state => state.projectsStatusText);
  
  // Select actions from store
  const setActiveTab = useAppStore(state => state.setActiveTab);
  const selectTopic = useAppStore(state => state.selectTopic);
  const loadSchema = useSchemaStore(state => state.loadSchema);

  // Load schema on mount
  useEffect(() => {
    loadSchema();
  }, [loadSchema]);

  // Refresh schema whenever the table tab becomes active
  useEffect(() => {
    if (activeTab === 'table') {
      loadSchema();
    }
  }, [activeTab, loadSchema]);

  // Generate mind map structure
  const mind = useMemo(() => {
    const taxonomy = Array.isArray(schema?.Taxonomy) ? schema.Taxonomy : [];
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
          children: (aspect.Options ?? []).map((opt, j) => ({ 
            id: `aspect-${i}-opt-${j}`, 
            topic: opt 
          }))
        }))
      }
    } as const;
  }, [schema]);

  const statusMessage = useMemo(() => {
    if (schemaStatus === 'loading') return 'Loading schema…';
    if (schemaStatus === 'error') return 'Unable to load schema';
    return 'Schema loaded';
  }, [schemaStatus]);

  return (
    <div className="page-shell">
      <div className="page-layout">
        <main className="mindmap-wrapper">
          <TabBar active={activeTab} onChange={setActiveTab} />
          <div className="tab-panels">
            <MindMap 
              active={activeTab === 'mindmap'} 
              mind={mind} 
              onSelect={selectTopic} 
              statusText={statusMessage} 
            />
            <SchemaTable 
              active={activeTab === 'table'} 
              schema={schema} 
              statusText={statusMessage} 
            />
          </div>
        </main>
        <ProjectPanel
          projects={projects}
          statusText={projectsStatusText}
          isLoading={projectsLoading}
        />
      </div>
    </div>
  );
}
