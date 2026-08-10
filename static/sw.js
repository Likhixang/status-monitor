/* status-monitor Service Worker：缓存静态资源支持离线，API 请求永不缓存（含 token）。 */
const CACHE = 'status-v1';
const SHELL = ['/', '/admin', '/manifest.json', '/icon-192.png', '/icon-512.png', '/icon-maskable-512.png', '/icon-180.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // 只处理同源 GET；API 请求走网络（带认证 token，绝不缓存）
  if (e.request.method !== 'GET' || url.origin !== self.location.origin || url.pathname.startsWith('/api/')) {
    return;
  }

  // 页面：网络优先，失败回退缓存（保证拿到最新 HTML）
  if (url.pathname === '/' || url.pathname === '/admin') {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(e.request, copy));
          return res;
        })
        .catch(() => caches.match(e.request).then((hit) => hit || caches.match('/')))
    );
    return;
  }

  // 静态资源（图标/manifest）：缓存优先
  e.respondWith(
    caches.match(e.request).then(
      (hit) => hit || fetch(e.request).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
    )
  );
});
