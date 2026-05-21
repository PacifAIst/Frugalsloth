/**
 * FrugalSloth Electron Main Process
 * Embeds the Vite-built app in a native window
 */

const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');

// Keep a global reference to prevent GC
let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    title: 'FrugalSloth — Edge AI Trainer',
    icon: path.join(__dirname, '../dist/icon.png'),
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
    },
    // Native feel
    titleBarStyle: 'default',
    show: false, // Show when ready
  });

  // Remove default menu for cleaner look (optional: uncomment to keep)
  Menu.setApplicationMenu(null);

  // Load the built Vite app
  const indexPath = path.join(__dirname, '../dist/index.html');
  mainWindow.loadFile(indexPath);

  // Show when ready to prevent white flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    // mainWindow.webContents.openDevTools(); // Uncomment for debugging
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Single instance lock
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
  process.exit(0);
}

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});
