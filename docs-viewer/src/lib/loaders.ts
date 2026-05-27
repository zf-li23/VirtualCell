// ⚠️ 此文件由 scripts/sync_notes_config.py 自动生成
// 手动修改将被覆盖！

import { noteMetas } from '../config/notes'

export type NoteLoader = () => Promise<string>

function fetchNote(path: string): NoteLoader {
  return () => fetch(`${import.meta.env.BASE_URL}notes/${path}`).then((r) => r.text())
}

export const noteLoaders: Record<string, NoteLoader> = {}
for (const meta of noteMetas) {
  noteLoaders[meta.id] = fetchNote(meta.path)
}
