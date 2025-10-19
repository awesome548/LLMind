import { useCallback } from 'react';


interface PromptConfig {
  systemPrompt: string;
  userPromptTemplate: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

interface TaxonomyNode {
  id: string;
  topic: string;
  parentid?: string;
  children?: TaxonomyNode[];
}

interface MindMapData {
  data: TaxonomyNode;
}

/**
 * Hook to interact with OpenAI using the current mind map taxonomy
 */
export function useOpenAITaxonomy(jmRef: React.RefObject<any>) {
  
  /**
   * Extract taxonomy from jsMind instance
   */
  const getTaxonomy = useCallback(() => {
    if (!jmRef.current) {
      console.error('jsMind instance not initialized');
      return null;
    }
    
    try {
      const mindData = jmRef.current.get_data('node_array').data;
      return mindData;
    } catch (error) {
      console.error('Error extracting taxonomy:', error);
      return null;
    }
  }, [jmRef]);

  /**
   * Load prompt configuration from JSON file
   */
  const loadPromptConfig = async (configPath: string): Promise<PromptConfig | null> => {
    try {
      const response = await fetch(configPath);
      if (!response.ok) {
        throw new Error(`Failed to load prompt config: ${response.statusText}`);
      }
      const config: PromptConfig = await response.json();
      return config;
    } catch (error) {
      console.error('Error loading prompt config:', error);
      return null;
    }
  };

  /**
   * Format taxonomy as a readable string for the prompt
   */
  const formatTaxonomy = useCallback((data: MindMapData): string => {
    const formatNode = (node: TaxonomyNode, indent: number = 0): string => {
      const prefix = '  '.repeat(indent);
      let result = `${prefix}- ${node.topic}\n`;
      
      if (node.children && node.children.length > 0) {
        node.children.forEach(child => {
          result += formatNode(child, indent + 1);
        });
      }
      
      return result;
    };

    return formatNode(data.data);
  }, []);

  /**
   * Call OpenAI API with taxonomy data
   */
  const callOpenAI = useCallback(async (
    apiKey: string,
    configPath: string = '/prompts/taxonomy-config.json'
  ): Promise<string | null> => {
    // Get current taxonomy
    const taxonomy = getTaxonomy();
    if (!taxonomy) {
      console.error('Failed to extract taxonomy');
      return null;
    }
    console.log('Extracted taxonomy:', taxonomy);

    // Load prompt configuration
    const config = await loadPromptConfig(configPath);
    if (!config) {
      console.error('Failed to load prompt configuration');
      return null;
    }

    // Format taxonomy for prompt
    const formattedTaxonomy = formatTaxonomy(taxonomy);
    
    // Replace placeholder in user prompt template
    const userPrompt = config.userPromptTemplate.replace(
      '{{TAXONOMY}}',
      formattedTaxonomy
    );

    // Prepare API request
    const requestBody = {
      model: config.model || 'gpt-4',
      messages: [
        {
          role: 'system',
          content: config.systemPrompt
        },
        {
          role: 'user',
          content: userPrompt
        }
      ],
      temperature: config.temperature ?? 0.7,
      max_tokens: config.maxTokens ?? 2000
    };

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`OpenAI API error: ${errorData.error?.message || response.statusText}`);
      }

      const data = await response.json();
      const result = data.choices?.[0]?.message?.content;
      
      if (!result) {
        throw new Error('No content in OpenAI response');
      }

      console.log('OpenAI response received successfully');
      return result;
      
    } catch (error) {
      console.error('Error calling OpenAI:', error);
      return null;
    }
  }, [getTaxonomy, formatTaxonomy]);

  return {
    getTaxonomy,
    callOpenAI,
    formatTaxonomy
  };
}