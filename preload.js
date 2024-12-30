const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    startCrawling: () => ipcRenderer.send('start-crawling'),
    onCrawlingResult: (callback) => ipcRenderer.on('crawling-result', (event, result) => callback(result))
});
