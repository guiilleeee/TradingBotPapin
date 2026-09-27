// Web Push for the TradingBot Papin dashboard (sent by the VPS, see web_push.py).

self.addEventListener("push", (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    data = { body: event.data ? event.data.text() : "" };
  }

  event.waitUntil(
    self.registration.showNotification(data.title || "TradingBot Papin", {
      body: data.body || "",
      icon: "icon-192x192.png",
      badge: "icon-192x192.png",
      vibrate: [200, 100, 200],
    })
  );
});

// Open the dashboard itself (this worker's scope, e.g. /TradingBotPapin/),
// reusing its tab if one is already open -- not the site root.
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const url = self.registration.scope;
  event.waitUntil(
    clients.matchAll({ type: "window", includeUncontrolled: true }).then((windows) => {
      for (const w of windows) {
        if (w.url.startsWith(url) && "focus" in w) return w.focus();
      }
      return clients.openWindow(url);
    })
  );
});
