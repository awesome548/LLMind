export const DEFAULT_TOPIC = 'Design Dimensions';

export type ProjectDetails = {
  id?: string;
  Id?: string;
  Name: string;
  Descriptions: string;
  Details: string;
  Image?: string;
};

export type SchemaOption = {
  name: string;
  desc?: string;
};

export type SchemaAspect = {
  name: string;
  desc?: string;
  options?: SchemaOption[];
};

export type SchemaDoc = {
  aspects: SchemaAspect[];
};
