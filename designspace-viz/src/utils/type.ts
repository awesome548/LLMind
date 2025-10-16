export const DEFAULT_TOPIC = 'Design Dimensions';

export type ProjectDetails = {
  id?: string;
  Id?: string;
  Name: string;
  Descriptions: string;
  Details: string;
  Image?: string;
};

export type SchemaAspect = {
  Aspect: string;
  Description?: string;
  Options?: string[];
};

export type SchemaDoc = {
  Taxonomy?: SchemaAspect[];
};
