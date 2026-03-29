import schemaData from '../../../../public/schema_selected.json';
import type { MindmapNode } from '../types';

const ROOT_ID = 'design-aspects';
const ROOT_TOPIC = 'Design Aspects';

function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

function buildNodes(): {
  nodes: ReadonlyArray<MindmapNode>;
  descriptionByTopic: Readonly<Record<string, string>>;
} {
  const descriptionByTopic: Record<string, string> = {
    [ROOT_TOPIC]: '',
  };

  const children: MindmapNode[] = schemaData.aspects.map((aspect) => {
    descriptionByTopic[aspect.name] = aspect.desc;

    const optionChildren: MindmapNode[] = aspect.options.map((option) => {
      descriptionByTopic[option.name] = option.desc;
      return { id: slugify(option.name), topic: option.name };
    });

    return {
      id: slugify(aspect.name),
      topic: aspect.name,
      children: optionChildren,
    };
  });

  const nodes: ReadonlyArray<MindmapNode> = [
    { id: ROOT_ID, topic: ROOT_TOPIC, children },
  ];

  return { nodes, descriptionByTopic: Object.freeze(descriptionByTopic) };
}

const { nodes, descriptionByTopic } = buildNodes();

export const SCHEMA_MINDMAP_NODES = nodes;
export const SCHEMA_DESCRIPTION_BY_TOPIC = descriptionByTopic;
