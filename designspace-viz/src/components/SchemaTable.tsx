import type { SchemaDoc } from '../types/taxonomy';
import { getSchemaAspects } from '../types/taxonomySchema';

export function SchemaTable({ active, schema, statusText }: { active: boolean; schema: SchemaDoc | null; statusText: string; }) {
  const taxonomy = getSchemaAspects(schema);
  return (
    <section id="table-panel" className={`tab-panel ${active ? 'active' : ''}`} role="tabpanel" aria-labelledby="table-tab" hidden={!active}>
      <div className="mindmap-header">
        <h1 className="mindmap-title">Schema Table View</h1>
        <p className="mindmap-subtitle">A quick scan of aspects with their available options.</p>
      </div>
      <div className="table-wrapper">
        <table className="schema-table" aria-describedby="table-status">
          <colgroup>
            <col className="schema-table__col schema-table__col--aspect" />
            <col className="schema-table__col schema-table__col--description" />
            <col className="schema-table__col schema-table__col--options" />
          </colgroup>
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
                <td className="schema-table__aspect">{aspect.name || `Aspect ${i + 1}`}</td>
                <td className="schema-table__description">{aspect.desc || '—'}</td>
                <td className="schema-table__options">
                  {(aspect.options?.length ? (
                    <ul>{aspect.options.map((option, j) => <li key={j}>{option.name}</li>)}</ul>
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
