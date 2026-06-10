import { useQuery } from '@tanstack/react-query';
import api from '@/src/lib/api-client';
import type {
  FetchRelatedProjectsRequestSchema,
  FetchRelatedProjectsResponseSchema,
} from '@/src/types/api-aliases';

const DEFAULT_LIMIT = 5;
const DEFAULT_SIMILARITY_THRESHOLD = 0.0;

const validateRequest = (
  request: FetchRelatedProjectsRequestSchema
): FetchRelatedProjectsRequestSchema => {
  const topic = request.topic.trim();
  if (!topic) {
    throw new Error('Failed to fetch related projects: topic is required.');
  }

  const requestedLimit = request.limit ?? DEFAULT_LIMIT;
  if (requestedLimit < 1 || requestedLimit > 20) {
    throw new Error('Failed to fetch related projects: limit must be 1-20.');
  }

  const similarityThreshold =
    request.similarity_threshold ?? DEFAULT_SIMILARITY_THRESHOLD;
  if (similarityThreshold < 0 || similarityThreshold > 1) {
    throw new Error(
      'Failed to fetch related projects: similarity_threshold must be 0-1.'
    );
  }

  return {
    ...request,
    topic,
    lineage: (request.lineage ?? []).filter(Boolean),
    limit: requestedLimit,
    similarity_threshold: similarityThreshold,
  };
};

export const fetchRelatedProjects = async (
  request: FetchRelatedProjectsRequestSchema
): Promise<FetchRelatedProjectsResponseSchema> => {
  const validatedRequest = validateRequest(request);

  try {
    const { data } = await api.post<FetchRelatedProjectsResponseSchema>(
      '/api/related-projects/search',
      validatedRequest
    );
    return {
      projects: (data.projects ?? []).map((project) => ({ ...project })),
    };
  } catch (error) {
    throw new Error('Failed to fetch related projects.', { cause: error });
  }
};

export interface UseRelatedProjectsQueryParams {
  request: FetchRelatedProjectsRequestSchema;
  enabled?: boolean;
}

export const useRelatedProjectsQuery = ({
  request,
  enabled = true,
}: UseRelatedProjectsQueryParams) =>
  useQuery({
    queryKey: [
      'mindmap-related-projects',
      request.topic,
      request.lineage ?? [],
      request.description ?? '',
      request.should_query_supabase ?? true,
      request.limit ?? DEFAULT_LIMIT,
      request.similarity_threshold ?? DEFAULT_SIMILARITY_THRESHOLD,
      request.embedding_model ?? '',
      request.match_function ?? '',
    ],
    queryFn: () => fetchRelatedProjects(request),
    enabled: enabled && Boolean(request.topic?.trim()),
    staleTime: 1000 * 60 * 5,
  });
