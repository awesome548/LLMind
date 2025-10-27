import OpenAI from 'openai';
import type { PromptConfig, TaxonomyNode, TaxonomyResponse, AddNodeResponse } from '../types/chatCompletion';

type FocusNodeContext = {
  id: string;
  topic: string;
};

type CallOpenAIOptions = {
  configPath: string;
  focusNode: FocusNodeContext | null;
};

/**
 * Hook to interact with OpenAI using the current mind map taxonomy
 */
export function useOpenAITaxonomy(jmRef: React.RefObject<any>) {
  
  /**
   * Extract taxonomy from jsMind instance
   */
  const getTaxonomy = () => {
    if (!jmRef.current) {
      console.error('jsMind instance not initialized');
      return null;
    }
    
    try {
      const mindData = jmRef.current.get_data('node_array');
      return mindData;
    } catch (error) {
      console.error('Error extracting taxonomy:', error);
      return null;
    }
  };

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
      if (config.userPromptTemplateFile) {
        const userResponse = await fetch(config.userPromptTemplateFile);
        if (userResponse.ok) {
          config.userPromptTemplate = await userResponse.text();
        }
      }
      return config;
    } catch (error) {
      console.error('Error loading prompt config:', error);
      return null;
    }
  };

  /**
   * Extract JSON from markdown code block
   */
  const extractJsonFromMarkdown = (text: string): TaxonomyResponse | null => {
    try {
      // Match content between ```json and ```
      const jsonMatch = text.match(/```json\s*([\s\S]*?)\s*```/);
      
      if (!jsonMatch || !jsonMatch[1]) {
        console.error('No JSON code block found in response');
        return null;
      }
      
      const jsonString = jsonMatch[1].trim();
      const parsed: TaxonomyResponse = JSON.parse(jsonString);
      
      return parsed;
    } catch (error) {
      console.error('Error extracting JSON from markdown:', error);
      return null;
    }
  };

  /**
   * Convert the parsed JSON to a flat array format
   */
  const convertToNodeArray = (jsonData: TaxonomyResponse): AddNodeResponse[] | null => {
    try {
      const parentId = jsonData.parent_id;
      const options = jsonData.options;
      
      if (!parentId || !options || typeof options !== 'object') {
        console.error('Invalid JSON structure: missing parent_id or options');
        return null;
      }
      
      const nodes: AddNodeResponse[] = [];
      
      // Convert options object to array of nodes
      Object.entries(options).forEach(([id, topic]) => {
        nodes.push({
            node_id: id,
            topic: topic as string,
            parent_node: parentId
        });
      });
      
      return nodes;
    } catch (error) {
      console.error('Error converting to node array:', error);
      return null;
    }
  };

  /**
   * Format taxonomy as a readable string for the prompt
   * Takes flat array from get_data().data and extracts only id and topic
   */
  const formatTaxonomy = (nodes: TaxonomyNode[]): string => {
    if (!Array.isArray(nodes) || nodes.length === 0) {
      return '';
    }

    // Create a map for quick parent lookup
    const nodeMap = new Map<string, TaxonomyNode>();
    nodes.forEach(node => nodeMap.set(node.id, node));

    // Find root node
    const root = nodes.find(node => node.isroot === true);
    if (!root) {
      console.error('No root node found');
      return '';
    }

    // Build hierarchy recursively
    const formatNode = (nodeId: string, indent: number = 0): string => {
      const node = nodeMap.get(nodeId);
      if (!node) return '';

      const prefix = '  '.repeat(indent);
      let result = `${prefix}- ${node.topic} (${node.id})\n`;

      // Find children (nodes whose parentid matches this node's id)
      const children = nodes.filter(n => n.parentid === nodeId);
      children.forEach(child => {
        result += formatNode(child.id, indent + 1);
      });

      return result;
    };

    return formatNode(root.id);
  };

  /**
   * Call OpenAI API with taxonomy data
   */
  const callOpenAI = async (
    apiKey: string,
    options: CallOpenAIOptions = {}
  ): Promise<AddNodeResponse[] | null> => {
    const configPath = options.configPath ?? '/prompts/system_prompt.json';
    const focusNode = options.focusNode ?? null;
    // Get current taxonomy
    const taxonomy = getTaxonomy();
    if (!taxonomy) {
      console.error('Failed to extract taxonomy');
      return null;
    }

    // Extract the data array from the taxonomy object
    const taxonomyData = taxonomy.data || taxonomy;
    if (!Array.isArray(taxonomyData)) {
      console.error('Taxonomy data is not an array');
      return null;
    }

    // Load prompt configuration
    const config = await loadPromptConfig(configPath);
    if (!config) {
      console.error('Failed to load prompt configuration');
      return null;
    }

    // Format taxonomy for prompt
    const formattedTaxonomy = formatTaxonomy(taxonomyData);
    
    const template = config.userPromptTemplate;

    // Replace placeholder in user prompt template
    let userPrompt = template.replace(
      '{{TAXONOMY}}',
      formattedTaxonomy
    );

    userPrompt = userPrompt
      .replaceAll('{{SELECTED_NODE_TOPIC}}', focusNode.topic)
      .replaceAll('{{SELECTED_NODE_ID}}', focusNode.id);

    console.info(userPrompt)

    const openai = new OpenAI({apiKey: apiKey, dangerouslyAllowBrowser: true});
    console.info('Calling OpenAI with formatted taxonomy');

    try {
      const completion = await openai.chat.completions.create({
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
      });


      const result = completion.choices?.[0]?.message?.content;
      
      if (!result) {
        throw new Error('No content in OpenAI response');
      }

      console.log('OpenAI response received successfully');
      console.log('Usage:', completion.usage);
      console.log('Raw response:', result);
      
      // Extract JSON from markdown code block
      const extractedJson = extractJsonFromMarkdown(result);
      if (!extractedJson) {
        console.error('Failed to extract JSON from response');
        return null;
      }
      
      // Convert to node array format
      const nodeArray = convertToNodeArray(extractedJson);
      if (!nodeArray) {
        console.error('Failed to convert to node array');
        return null;
      }
      
      console.log('Parsed node array:', nodeArray);
      return nodeArray;

      
    } catch (error) {
      console.error('Error calling OpenAI:', error);
      return null;
    }
  };

  return {
    getTaxonomy,
    callOpenAI,
    formatTaxonomy
  };
}
