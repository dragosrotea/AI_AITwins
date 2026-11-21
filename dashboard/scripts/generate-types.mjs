import { readFile, writeFile } from 'node:fs/promises'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const projectRoot = resolve(scriptDir, '../..')
const contractPath = resolve(projectRoot, 'docs/contract.json')
const outputPath = resolve(scriptDir, '../src/types.ts')
const contract = JSON.parse(await readFile(contractPath, 'utf8'))

function typeFor(schema) {
  if (schema.$ref) return schema.$ref.split('/').at(-1)
  if (schema.anyOf) return schema.anyOf.map(typeFor).join(' | ')
  if (schema.enum) return schema.enum.map((value) => JSON.stringify(value)).join(' | ')
  if (schema.const !== undefined) return JSON.stringify(schema.const)

  switch (schema.type) {
    case 'string': return 'string'
    case 'number':
    case 'integer': return 'number'
    case 'boolean': return 'boolean'
    case 'null': return 'null'
    case 'array': return `Array<${typeFor(schema.items ?? {})}>`
    case 'object': {
      if (!schema.properties) return 'Record<string, unknown>'
      const required = new Set(schema.required ?? [])
      const fields = Object.entries(schema.properties).map(([name, property]) =>
        `  ${JSON.stringify(name)}${required.has(name) ? '' : '?'}: ${typeFor(property)};`,
      )
      return `{\n${fields.join('\n')}\n}`
    }
    default: return 'unknown'
  }
}

const definitions = Object.entries(contract.$defs ?? {}).filter(([name]) => name !== 'Envelope').map(([name, schema]) => {
  if (schema.type === 'object' && schema.properties) {
    const required = new Set(schema.required ?? [])
    const fields = Object.entries(schema.properties).map(([field, property]) =>
      `  ${JSON.stringify(field)}${required.has(field) ? '' : '?'}: ${typeFor(property)};`,
    )
    return `export interface ${name} {\n${fields.join('\n')}\n}`
  }
  return `export type ${name} = ${typeFor(schema)}`
})

const topics = Object.keys(contract.topics ?? {})
const sourceType = typeFor(contract.$defs.Envelope.properties.src)
const topicRows = topics.map((topic) => {
  const model = contract.topics[topic].model.$ref.split('/').at(-1)
  return `  ${JSON.stringify(topic)}: ${model};`
})

const output = `/* Generated from docs/contract.json. Do not edit by hand. */

${definitions.join('\n\n')}

export const TOPICS = ${JSON.stringify(topics, null, 2)} as const
export type Topic = (typeof TOPICS)[number]

export interface TopicDataMap {
${topicRows.join('\n')}
}

export type Src = ${sourceType}

export interface Envelope<TData = Record<string, unknown>> {
  topic: string
  ts: number
  src: Src
  data: TData
}

export type TopicEnvelope<K extends Topic = Topic> = Omit<Envelope<TopicDataMap[K]>, 'topic'> & {
  topic: K
}

export type AnyTopicEnvelope = { [K in Topic]: TopicEnvelope<K> }[Topic]
`

await writeFile(outputPath, output)
console.log(`Generated ${outputPath} from ${contractPath}`)
