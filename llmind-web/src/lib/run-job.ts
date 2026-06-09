import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';

interface JobStatus<T> {
  status: 'pending' | 'done' | 'error';
  result: T | null;
  detail: string | null;
}

interface RunJobOptions {
  intervalMs?: number;
  timeoutMs?: number;
  signal?: AbortSignal;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Start a long-running backend job and poll it to completion.
 *
 * The start endpoint returns `{ job_id }` immediately; we then poll
 * `GET /api/jobs/{id}` with short requests until the job is `done` or `error`.
 * This keeps every HTTP request short (no 50-80s held connection) and lets the
 * caller drive progress UI off the returned promise.
 */
export async function runJob<T>(
  startPath: string,
  body: unknown,
  options: RunJobOptions = {}
): Promise<T> {
  const intervalMs = options.intervalMs ?? 1500;
  const timeoutMs = options.timeoutMs ?? 5 * 60 * 1000;

  let jobId: string;
  try {
    const { data } = await api.post<{ job_id: string }>(startPath, body, {
      signal: options.signal,
    });
    jobId = data.job_id;
  } catch (error) {
    throw toError(error, 'Failed to start generation');
  }

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    await sleep(intervalMs);
    if (options.signal?.aborted) throw new Error('Generation cancelled.');

    let status: JobStatus<T>;
    try {
      const { data } = await api.get<JobStatus<T>>(`/api/jobs/${jobId}`, {
        signal: options.signal,
      });
      status = data;
    } catch (error) {
      // A transient poll failure shouldn't abort the whole job — retry next tick.
      if (isAxiosError(error) && error.response?.status === 404) {
        throw new Error('Generation job expired or was lost.');
      }
      continue;
    }

    if (status.status === 'done') return status.result as T;
    if (status.status === 'error') throw new Error(status.detail ?? 'Generation failed.');
  }
  throw new Error('Generation timed out.');
}

function toError(error: unknown, prefix: string): Error {
  if (isAxiosError(error)) {
    const detail =
      error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
        ? String((error.response.data as { detail: unknown }).detail)
        : error.message;
    return new Error(`${prefix}: ${detail}`);
  }
  return new Error(`${prefix}.`);
}
