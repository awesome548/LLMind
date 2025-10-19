export interface PromptConfig {
  userPromptTemplateFile?: string;
  systemPrompt: string;
  userPromptTemplate: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}


export interface TaxonomyNode {
  id: string;
  topic: string;
  parentid?: string;
  expanded?: boolean;
  isroot?: boolean;
  direction?: string;
  description?: string;
}

export interface TaxonomyResponse {
  parent_id: string;
  options: { [key: string]: string };
}

export interface AddNodeResponse {
  node_id: string;
  topic: string;
  parent_node: string;
}