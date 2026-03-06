import { z } from 'zod';
import type { SchemaAspect, SchemaDoc } from './taxonomy';

const emptyStringToUndefined = (value: unknown): unknown => {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  return trimmed.length ? trimmed : undefined;
};

const optionalTextSchema = z.preprocess(
  emptyStringToUndefined,
  z.string().trim().min(1).optional()
);

export const schemaOptionSchema = z.object({
  name: z.string().trim().min(1),
  desc: optionalTextSchema,
});

export const schemaAspectSchema = z.object({
  name: z.string().trim().min(1),
  desc: optionalTextSchema,
  options: z.array(schemaOptionSchema).optional(),
});

export const schemaDocSchema = z.object({
  aspects: z.array(schemaAspectSchema),
});

const canonicalOptionInputSchema = z.object({
  name: z.string().trim().min(1),
  desc: optionalTextSchema,
}).strict();

const canonicalAspectInputSchema = z.object({
  name: z.string().trim().min(1),
  desc: optionalTextSchema,
  options: z.array(canonicalOptionInputSchema).optional(),
}).strict();

const canonicalDocInputSchema = z.object({
  aspects: z.array(canonicalAspectInputSchema),
}).strict();

const formatIssuePath = (path: (string | number)[]): string => {
  if (!path.length) {
    return 'payload';
  }

  return path.reduce<string>((acc, segment) => {
    if (typeof segment === 'number') {
      return `${acc}[${segment}]`;
    }
    if (!acc) {
      return `${segment}`;
    }
    return `${acc}.${segment}`;
  }, '');
};

const formatZodError = (error: z.ZodError): string =>
  error.issues
    .map((issue) => `${formatIssuePath(issue.path)}: ${issue.message}`)
    .join('; ');

const parseCanonicalDocBranch = (input: unknown): SchemaDoc => {
  const parsed = canonicalDocInputSchema.safeParse(input);
  if (!parsed.success) {
    throw new Error(formatZodError(parsed.error));
  }

  const schema = schemaDocSchema.safeParse(parsed.data);
  if (!schema.success) {
    throw new Error(formatZodError(schema.error));
  }

  return schema.data as SchemaDoc;
};

export const parseSchemaDoc = (input: unknown): SchemaDoc => {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    throw new Error('payload: Expected an object with canonical shape { aspects: [...] }.');
  }

  const candidate = input as Record<string, unknown>;
  if ('Taxonomy' in candidate) {
    throw new Error(
      'payload: Legacy key "Taxonomy" is not supported. Use canonical key "aspects".'
    );
  }
  if (!('aspects' in candidate)) {
    throw new Error('payload: Missing required key "aspects".');
  }

  return parseCanonicalDocBranch(input);
};

export const safeParseSchemaDoc = (
  input: unknown
): { success: true; data: SchemaDoc } | { success: false; error: string } => {
  try {
    const data = parseSchemaDoc(input);
    return { success: true, data };
  } catch (error) {
    if (error instanceof Error) {
      return { success: false, error: error.message };
    }
    return { success: false, error: 'Unknown schema parsing error.' };
  }
};

export const getSchemaAspects = (schema: SchemaDoc | null | undefined): SchemaAspect[] => {
  if (!schema) {
    return [];
  }
  return schema.aspects;
};
