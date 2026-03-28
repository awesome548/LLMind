import type { MindmapProjectSchema } from '@/src/types/openapi';

type LooseProjectRecord = MindmapProjectSchema & Record<string, unknown>;

const ABSOLUTE_PROTOCOLS = new Set(['http:', 'https:']);

function readString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : '';
}

export function getProjectId(project: MindmapProjectSchema, index: number): string {
  const record = project as LooseProjectRecord;
  return readString(record.id) || readString(record.Id) || `project-${index}`;
}

export function getProjectName(project: MindmapProjectSchema): string {
  const record = project as LooseProjectRecord;
  return readString(record.Name) || readString(record.name) || 'Untitled';
}

export function getProjectDescription(project: MindmapProjectSchema): string {
  const record = project as LooseProjectRecord;
  return readString(record.Descriptions) || readString(record.description);
}

export function getProjectDetail(project: MindmapProjectSchema): string {
  const record = project as LooseProjectRecord;
  return readString(record.Details) || readString(record.detail);
}

export function getProjectImageUrl(project: MindmapProjectSchema): string | null {
  const record = project as LooseProjectRecord;
  const rawValue = readString(record.Image) || readString(record.image);
  if (!rawValue) {
    return null;
  }

  if (rawValue.startsWith('/')) {
    return rawValue;
  }

  try {
    const parsedUrl = new URL(rawValue);
    return ABSOLUTE_PROTOCOLS.has(parsedUrl.protocol) ? parsedUrl.toString() : null;
  } catch {
    return null;
  }
}
