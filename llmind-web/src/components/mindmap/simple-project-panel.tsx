'use client';

import { useState } from 'react';
import {
  getProjectDescription,
  getProjectDetail,
  getProjectId,
  getProjectImageUrl,
  getProjectName,
} from '@/src/components/mindmap/project-fields';
import { cn } from '@/src/lib/utils';
import type { MindmapProjectSchema } from '@/src/types/openapi';

interface SimpleProjectPanelProps {
  projects: ReadonlyArray<MindmapProjectSchema>;
  isLoading?: boolean;
}

function projectKey(project: MindmapProjectSchema, index: number): string {
  return getProjectId(project, index);
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max).trimEnd()}…`;
}

function Skeleton() {
  return (
    <div className="space-y-2 animate-pulse" aria-busy aria-label="Loading projects">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className="h-14 rounded-lg bg-muted" />
      ))}
    </div>
  );
}

export function SimpleProjectPanel({ projects, isLoading = false }: SimpleProjectPanelProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const items = projects.map((p, i) => ({ id: projectKey(p, i), project: p }));
  const activeId = items.some((it) => it.id === selectedId)
    ? selectedId
    : (items[0]?.id ?? null);
  const selected = items.find((it) => it.id === activeId)?.project ?? null;

  return (
    <div className="flex min-h-[520px] flex-col overflow-hidden rounded-xl border bg-card">
      <div className="border-b px-4 py-3">
        <p className="text-sm font-semibold">Related Projects</p>
      </div>

      <div className="grid flex-1 grid-cols-[minmax(0,2fr)_minmax(0,3fr)] divide-x overflow-hidden">
        {/* List */}
        <div className="overflow-y-auto p-3">
          {isLoading ? (
            <Skeleton />
          ) : items.length === 0 ? (
            <p className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground">
              No projects found.
            </p>
          ) : (
            <ul className="space-y-1.5">
              {items.map((item) => (
                <li key={item.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(item.id)}
                    className={cn(
                      'w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-colors',
                      item.id === activeId
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border hover:bg-muted'
                    )}
                  >
                    <p className="font-medium leading-snug">
                      {getProjectName(item.project)}
                    </p>
                    {getProjectDescription(item.project) ? (
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {truncate(getProjectDescription(item.project), 70)}
                      </p>
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Detail */}
        <div className="overflow-y-auto p-4">
          {isLoading ? (
            <div className="space-y-3 animate-pulse">
              <div className="h-5 w-2/3 rounded bg-muted" />
              <div className="h-32 rounded-lg bg-muted" />
              <div className="h-4 rounded bg-muted" />
              <div className="h-4 w-4/5 rounded bg-muted" />
            </div>
          ) : selected ? (
            <div className="space-y-3">
              <h3 className="font-semibold leading-tight">{getProjectName(selected)}</h3>
              {getProjectImageUrl(selected) ? (
                <img
                  src={getProjectImageUrl(selected) ?? ''}
                  alt={getProjectName(selected) || 'Project image'}
                  className="h-36 w-full rounded-lg object-cover"
                  loading="lazy"
                  referrerPolicy="no-referrer"
                />
              ) : null}
              {getProjectDescription(selected) ? (
                <p className="text-sm text-muted-foreground">
                  {getProjectDescription(selected)}
                </p>
              ) : null}
              {getProjectDetail(selected) ? (
                <p className="text-sm">{getProjectDetail(selected)}</p>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Select a project to view details.</p>
          )}
        </div>
      </div>
    </div>
  );
}
