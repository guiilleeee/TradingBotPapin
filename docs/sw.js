self.addEventListener("push", (event) => {
  const data = event.data ? event.data.json() : {};
  
  const options = {
    body: data.body || "TradingBot Update",
    icon: "icon-192x192.png",
    badge: "icon-192x192.png",
    vibrate: [200, 100, 200],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1,
    }
  };

  event.waitUntil(
    self.registration.showNotification(data.title || "TradingBot", options)
  );
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow("/")
  );
});
