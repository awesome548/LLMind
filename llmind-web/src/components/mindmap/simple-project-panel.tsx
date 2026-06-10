'use client';

import { useState } from 'react';
import { cn } from '@/src/lib/utils';
import type { MindmapProjectSchema } from '@/src/types/api-aliases';

interface SimpleProjectPanelProps {
  projects: ReadonlyArray<MindmapProjectSchema>;
  isLoading?: boolean;
  /** A corpus project opened from the design space (or a provenance chip) —
   * shown at the top of the list and auto-selected when it changes. */
  focusProject?: MindmapProjectSchema | null;
}

type LooseProjectRecord = MindmapProjectSchema & Record<string, unknown>;

interface ProjectListItemViewModel {
  id: string;
  name: string;
  description: string;
  detail: string;
  imageUrl: string | null;
}

function readString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

function getProjectId(project: LooseProjectRecord, index: number): string {
  return readString(project.id) || readString(project.Id) || `project-${index}`;
}

function getProjectName(project: LooseProjectRecord): string {
  return readString(project.Name) || readString(project.name) || 'Untitled';
}

function getProjectDescription(project: LooseProjectRecord): string {
  return readString(project.Descriptions) || readString(project.description);
}

function getProjectDetail(project: LooseProjectRecord): string {
  return readString(project.Details) || readString(project.detail);
}

function getProjectImageUrl(project: LooseProjectRecord): string | null {
  const rawValue = readString(project.Image) || readString(project.image);
  if (!rawValue) {
    return null;
  }

  if (rawValue.startsWith('/')) {
    return rawValue;
  }

  try {
    const parsedUrl = new URL(rawValue);
    return parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:'
      ? parsedUrl.toString()
      : null;
  } catch {
    return null;
  }
}

function toProjectListItem(
  project: MindmapProjectSchema,
  index: number
): ProjectListItemViewModel {
  const record = project as LooseProjectRecord;

  return {
    id: getProjectId(record, index),
    name: getProjectName(record),
    description: getProjectDescription(record),
    detail: getProjectDetail(record),
    imageUrl: getProjectImageUrl(record),
  };
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

interface ProjectListProps {
  items: ReadonlyArray<ProjectListItemViewModel>;
  activeId: string | null;
  onSelect: (id: string) => void;
}

function ProjectList({ items, activeId, onSelect }: ProjectListProps) {
  if (items.length === 0) {
    return (
      <p className="rounded-lg border border-dashed p-3 text-sm text-muted-foreground">
        No projects found.
      </p>
    );
  }

  return (
    <ul className="space-y-1.5">
      {items.map((item) => (
        <li key={item.id}>
          <button
            type="button"
            onClick={() => onSelect(item.id)}
            className={cn(
              'w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-colors',
              item.id === activeId
                ? 'border-primary bg-primary/10 text-primary'
                : 'border-border hover:bg-muted'
            )}
          >
            <p className="font-medium leading-snug">{item.name}</p>
            {item.description ? (
              <p className="mt-0.5 text-xs text-muted-foreground">
                {truncate(item.description, 70)}
              </p>
            ) : null}
          </button>
        </li>
      ))}
    </ul>
  );
}

interface ProjectDetailProps {
  project: ProjectListItemViewModel | null;
}

function ProjectDetail({ project }: ProjectDetailProps) {
  if (!project) {
    return <p className="text-sm text-muted-foreground">Select a project to view details.</p>;
  }

  return (
    <div className="space-y-3">
      <h3 className="font-semibold leading-tight">{project.name}</h3>
      {project.imageUrl ? (
        <img
          key={project.imageUrl}
          src={project.imageUrl}
          alt={project.name || 'Project image'}
          className="h-36 w-full rounded-lg object-cover"
          loading="lazy"
          referrerPolicy="no-referrer"
        />
      ) : null}
      {project.description ? (
        <p className="text-sm text-muted-foreground">{project.description}</p>
      ) : null}
      {project.detail ? <p className="text-sm">{project.detail}</p> : null}
    </div>
  );
}

export function SimpleProjectPanel({
  projects,
  isLoading = false,
  focusProject = null,
}: SimpleProjectPanelProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const focusItem = focusProject ? toProjectListItem(focusProject, -1) : null;
  // Auto-select the focused project whenever a new one is opened (state adjusted
  // during render — no effect, no cascading re-render).
  const focusId = focusItem?.id ?? null;
  const [lastFocusId, setLastFocusId] = useState<string | null>(null);
  if (focusId !== lastFocusId) {
    setLastFocusId(focusId);
    if (focusId) setSelectedId(focusId);
  }

  const related = projects.map(toProjectListItem);
  const items =
    focusItem && !related.some((item) => item.id === focusItem.id)
      ? [focusItem, ...related]
      : related;
  const activeId = items.some((item) => item.id === selectedId)
    ? selectedId
    : (items[0]?.id ?? null);
  const selectedProject = items.find((item) => item.id === activeId) ?? null;

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
          ) : (
            <ProjectList items={items} activeId={activeId} onSelect={setSelectedId} />
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
          ) : (
            <ProjectDetail project={selectedProject} />
          )}
        </div>
      </div>
    </div>
  );
}
