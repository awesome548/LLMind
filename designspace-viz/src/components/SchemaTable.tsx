import type { SchemaDoc } from '../utils/type';

export function SchemaTable({ active, schema, statusText }: { active: boolean; schema: SchemaDoc | null; statusText: string; }) {
  const taxonomy = Array.isArray(schema?.Taxonomy) ? schema!.Taxonomy! : [];
  return (
    <section id="table-panel" className={`tab-panel ${active ? 'active' : ''}`} role="tabpanel" aria-labelledby="table-tab" hidden={!active}>
      <div className="mindmap-header">
        <h1 className="mindmap-title">Schema Table View</h1>
        <p className="mindmap-subtitle">A quick scan of aspects with their available options.</p>
      </div>
      <div className="table-wrapper">
        <table className="schema-table" aria-describedby="table-status">
          <thead>
            <tr>
              <th scope="col">Aspect</th>
              <th scope="col">Description</th>
              <th scope="col">Options</th>
            </tr>
          </thead>
          <tbody id="schema-table-body">
            {taxonomy.length === 0 ? (
              <tr><td colSpan={3}>No aspects available in schema.</td></tr>
            ) : taxonomy.map((aspect, i) => (
              <tr key={i}>
                <td>{aspect.Aspect || `Aspect ${i + 1}`}</td>
                <td>{aspect.Description || '—'}</td>
                <td>
                  {(aspect.Options?.length ? (
                    <ul>{aspect.Options!.map((o, j) => <li key={j}>{o}</li>)}</ul>
                  ) : '—')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <span id="table-status" className="status">{statusText}</span>
    </section>
  );
}
