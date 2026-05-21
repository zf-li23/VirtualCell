import { noteMetas } from '../config/notes'

export type NoteLoader = () => Promise<string>

function fetchNote(path: string): NoteLoader {
  return () => fetch(`${import.meta.env.BASE_URL}notes/${path}`).then((r) => r.text())
}

/**
 * 根据 noteMetas 中的 path 字段自动生成加载器。
 * 不再需要手动维护第二个配置文件。
 */
export const noteLoaders: Record<string, NoteLoader> = {}
for (const meta of noteMetas) {
  noteLoaders[meta.id] = fetchNote(meta.path)
}
