type TabId = 'mindmap' | 'table';
type Tab = { id: TabId; label: string };

export function TabBar({ active, onChange }: { active: TabId; onChange: (id: TabId) => void }) {
  const tabs: Tab[] = [
    { id: 'mindmap', label: 'Mind Map' },
    { id: 'table', label: 'Table' },
  ];
  return (
    <div className="tab-bar" role="tablist" aria-label="Schema views">
      {tabs.map(t => (
        <button
          key={t.id}
          type="button"
          id={`${t.id}-tab`}
          className={`tab-button ${active === t.id ? 'active' : ''}`}
          role="tab"
          aria-selected={active === t.id}
          aria-controls={`${t.id}-panel`}
          onClick={() => onChange(t.id)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
