#!/usr/bin/env node

/**
 * sync-notes.mjs
 * 
 * 将项目根目录 notes/ 同步到 docs-viewer/public/notes/
 * 确保 docs-viewer 构建时使用最新的笔记内容。
 * 
 * 用法:
 *   node scripts/sync-notes.mjs          # 一次性同步
 *   node scripts/sync-notes.mjs --watch  # 持续监听变化
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(__dirname, '..', '..')
const viewerRoot = path.resolve(__dirname, '..')

const SRC = path.join(projectRoot, 'notes')
const DEST = path.join(viewerRoot, 'public', 'notes')

/** 递归拷贝目录，只覆盖已存在的文件，跳过 dest 中多余的文件 */
function syncDir(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true })
  }

  let copiedCount = 0
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name)
    const destPath = path.join(dest, entry.name)

    if (entry.isDirectory()) {
      copiedCount += syncDir(srcPath, destPath)
    } else {
      const srcStat = fs.statSync(srcPath)
      let needsCopy = true

      // 如果目标已存在，比较 mtime 决定是否真的需要拷贝
      if (fs.existsSync(destPath)) {
        const destStat = fs.statSync(destPath)
        if (destStat.mtimeMs >= srcStat.mtimeMs && destStat.size === srcStat.size) {
          needsCopy = false
        }
      }

      if (needsCopy) {
        fs.mkdirSync(path.dirname(destPath), { recursive: true })
        fs.copyFileSync(srcPath, destPath)
        // 保留原始文件的 mtime，方便后续比较
        fs.utimesSync(destPath, srcStat.atime, srcStat.mtime)
        copiedCount++
      }
    }
  }

  return copiedCount
}

function main() {
  const watchMode = process.argv.includes('--watch')

  // 检测是否已是符号链接 → 无需手动同步
  try {
    if (fs.lstatSync(DEST).isSymbolicLink()) {
      const linkTarget = fs.readlinkSync(DEST)
      const resolvedDest = path.resolve(path.dirname(DEST), linkTarget)
      const resolvedSrc = path.resolve(SRC)
      if (resolvedDest === resolvedSrc) {
        console.log(`[sync-notes] ✅ public/notes 已是符号链接 (-> ${linkTarget})，无需同步。`)
        return
      }
    }
  } catch {
    // DEST 不存在或无法访问 → 继续执行同步
  }

  console.log(`[sync-notes] Syncing notes from:\n  ${SRC}\n  → ${DEST}`)

  const count = syncDir(SRC, DEST)
  console.log(`[sync-notes] Done. ${count} file(s) copied/updated.`)

  if (watchMode) {
    console.log('[sync-notes] Watching for changes...')
    // 使用 fs.watch 监听源目录变化
    const debounceTimers = new Map()

    function onChange(eventType, filename) {
      if (!filename) return

      const srcPath = path.join(SRC, filename)
      const destPath = path.join(DEST, filename)

      // 清除之前的 debounce 定时器
      const key = filename
      if (debounceTimers.has(key)) {
        clearTimeout(debounceTimers.get(key))
      }

      debounceTimers.set(key, setTimeout(() => {
        debounceTimers.delete(key)
        try {
          if (fs.existsSync(srcPath)) {
            const stat = fs.statSync(srcPath)
            if (stat.isDirectory()) {
              syncDir(srcPath, destPath)
            } else {
              fs.mkdirSync(path.dirname(destPath), { recursive: true })
              fs.copyFileSync(srcPath, destPath)
              fs.utimesSync(destPath, stat.atime, stat.mtime)
            }
            console.log(`[sync-notes] Updated: ${filename}`)
          } else {
            // 源文件被删除 → 不再自动删除 dest 中的对应文件
            // （避免误删手动添加的文件）
            console.log(`[sync-notes] Source removed (skipped): ${filename}`)
          }
        } catch (err) {
          console.error(`[sync-notes] Error syncing ${filename}:`, err.message)
        }
      }, 100))
    }

    // 监听整个 notes 目录树
    function watchDir(dir) {
      try {
        fs.watch(dir, { recursive: true }, onChange)
      } catch {
        // 递归监听在某些系统上可能不支持
        fs.watch(dir, onChange)
        for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
          if (entry.isDirectory()) {
            watchDir(path.join(dir, entry.name))
          }
        }
      }
    }

    watchDir(SRC)
  }
}

main()
