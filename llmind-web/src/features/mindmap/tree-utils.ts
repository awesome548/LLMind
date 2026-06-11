// Pure tree operations on the working MindmapNode tree. Extracted from the
// page so they are unit-testable and reusable; no React, no store.

import type { MindmapNode } from './types';

export interface TreeUpdateResult {
  nodes: ReadonlyArray<MindmapNode>;
  inserted: boolean;
}

/** Walk a topic path from the roots; null when any segment is missing. */
export function findNodeByLineage(
  nodes: ReadonlyArray<MindmapNode>,
  lineage: ReadonlyArray<string>
): MindmapNode | null {
  let currentNodes = nodes;
  let currentNode: MindmapNode | undefined;

  for (const topic of lineage) {
    currentNode = currentNodes.find((node) => node.topic === topic);
    if (!currentNode) {
      return null;
    }
    currentNodes = currentNode.children ?? [];
  }

  return currentNode ?? null;
}

/** Immutably append children under `parentId` (existing ids are skipped). */
export function insertChildrenAtNode(
  nodes: ReadonlyArray<MindmapNode>,
  parentId: string,
  childrenToInsert: ReadonlyArray<MindmapNode>
): TreeUpdateResult {
  let inserted = false;
  const nextNodes = nodes.map((node) => {
    if (node.id === parentId) {
      inserted = true;
      const existingChildren = node.children ?? [];
      const existingIds = new Set(existingChildren.map((child) => child.id));
      const uniqueNewChildren = childrenToInsert.filter((child) => !existingIds.has(child.id));
      return {
        ...node,
        children: [...existingChildren, ...uniqueNewChildren],
      };
    }

    if (!node.children?.length) {
      return node;
    }

    const childResult = insertChildrenAtNode(node.children, parentId, childrenToInsert);
    if (!childResult.inserted) {
      return node;
    }

    inserted = true;
    return {
      ...node,
      children: childResult.nodes,
    };
  });

  return { nodes: nextNodes, inserted };
}

/** Collect every node id in the tree into `into`. */
export function collectIds(nodes: ReadonlyArray<MindmapNode>, into: Set<string>): void {
  for (const node of nodes) {
    into.add(node.id);
    if (node.children?.length) collectIds(node.children, into);
  }
}

/**
 * Node ids are slugified names (and LLM-supplied ids for generated nodes), so a
 * newly generated option can collide with an existing node anywhere in the tree
 * — duplicate ids break React keys and the id→coordinate mapping in the design
 * space. Remap any colliding child id to a unique one and report the remap so
 * callers can keep coordinates aligned.
 */
export function ensureUniqueChildIds(
  allNodes: ReadonlyArray<MindmapNode>,
  children: ReadonlyArray<MindmapNode>
): { children: MindmapNode[]; remap: Record<string, string> } {
  const used = new Set<string>();
  collectIds(allNodes, used);
  const remap: Record<string, string> = {};

  const result = children.map((child) => {
    let id = child.id;
    if (used.has(id)) {
      let n = 2;
      while (used.has(`${child.id}-${n}`)) n++;
      id = `${child.id}-${n}`;
      remap[child.id] = id;
    }
    used.add(id);
    return { ...child, id };
  });

  return { children: result, remap };
}
