const { app, BrowserWindow, dialog, Menu, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { exec } = require('child_process');

let mainWindow;
let downloadDirectory = null;
let researchDirectory = null;

// Settings management
const loadSettings = () => {
    const settingsPath = path.join(app.getPath('userData'), 'settings.json');
    if (fs.existsSync(settingsPath)) {
        const settings = JSON.parse(fs.readFileSync(settingsPath));
        downloadDirectory = settings.downloadDirectory;
        researchDirectory = settings.researchDirectory;
        return true;
    }
    return false;
};

const saveSettings = () => {
    const settings = {
        downloadDirectory,
        researchDirectory
    };
    fs.writeFileSync(
        path.join(app.getPath('userData'), 'settings.json'),
        JSON.stringify(settings)
    );
};

const selectDirectory = async (title) => {
    const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
        title,
        properties: ['openDirectory']
    });
    if (canceled) return null;
    return filePaths[0];
};

const confirmDirectory = async (directory, message) => {
    const result = await dialog.showMessageBox(mainWindow, {
        type: 'question',
        buttons: ['Confirm', 'Select Again'],
        title: 'Confirm Directory',
        message: `${message}:\n${directory}\n\nIs this correct?`
    });
    return result.response === 0; // 0 = Confirm
};

const setupDownloadDirectory = async () => {
    let confirmed = false;
    while (!confirmed) {
        await dialog.showMessageBox(mainWindow, {
            type: 'info',
            buttons: ['OK'],
            title: 'Download Directory Setup',
            message: 'Please select a directory where downloaded files will be saved.\n\nThis directory will be used to store temporary files during the crawling process.'
        });

        const dir = await selectDirectory('Select Download Directory');
        if (!dir) {
            const useDefault = await dialog.showMessageBox(mainWindow, {
                type: 'question',
                buttons: ['Use Default', 'Select Again'],
                title: 'Use Default Directory?',
                message: `Would you like to use the default download directory?\n${path.join(app.getPath('downloads'), 'Crawl4AI')}`
            });
            if (useDefault.response === 0) {
                downloadDirectory = path.join(app.getPath('downloads'), 'Crawl4AI');
                confirmed = true;
            }
        } else {
            confirmed = await confirmDirectory(dir, 'Selected download directory');
            if (confirmed) {
                downloadDirectory = dir;
            }
        }
    }
    fs.mkdirSync(downloadDirectory, { recursive: true });
};

const setupResearchDirectory = async () => {
    let confirmed = false;
    while (!confirmed) {
        await dialog.showMessageBox(mainWindow, {
            type: 'info',
            buttons: ['OK'],
            title: 'Research Directory Setup',
            message: 'Please select a directory where processed research files will be saved.\n\nThis directory will store the final processed and organized research materials.'
        });

        const dir = await selectDirectory('Select Research Directory');
        if (!dir) {
            const useDefault = await dialog.showMessageBox(mainWindow, {
                type: 'question',
                buttons: ['Use Default', 'Select Again'],
                title: 'Use Default Directory?',
                message: `Would you like to use the default research directory?\n${path.join(app.getPath('documents'), 'Crawl4AI Research')}`
            });
            if (useDefault.response === 0) {
                researchDirectory = path.join(app.getPath('documents'), 'Crawl4AI Research');
                confirmed = true;
            }
        } else {
            confirmed = await confirmDirectory(dir, 'Selected research directory');
            if (confirmed) {
                researchDirectory = dir;
            }
        }
    }
    fs.mkdirSync(researchDirectory, { recursive: true });
};

const setupDirectories = async () => {
    await dialog.showMessageBox(mainWindow, {
        type: 'info',
        buttons: ['OK'],
        title: 'First Time Setup',
        message: 'Welcome to Crawl4AI!\n\nLet\'s set up your directories for downloads and research materials.'
    });

    await setupDownloadDirectory();
    await setupResearchDirectory();

    saveSettings();

    await dialog.showMessageBox(mainWindow, {
        type: 'info',
        buttons: ['OK'],
        title: 'Setup Complete',
        message: `Setup complete! Your directories have been configured:\n\nDownloads: ${downloadDirectory}\nResearch: ${researchDirectory}\n\nYou can change these directories anytime from the File menu.`
    });
};

// Create application menu
const createMenu = () => {
    const menuTemplate = [
        {
            label: 'File',
            submenu: [
                {
                    label: 'Change Download Directory',
                    click: async () => {
                        await setupDownloadDirectory();
                        saveSettings();
                        // Restart Docker container to apply new settings
                        await stopDockerService();
                        await startDockerService();
                    }
                },
                {
                    label: 'Change Research Directory',
                    click: async () => {
                        await setupResearchDirectory();
                        saveSettings();
                        // Restart Docker container to apply new settings
                        await stopDockerService();
                        await startDockerService();
                    }
                },
                {
                    label: 'Show Current Directories',
                    click: async () => {
                        await dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            buttons: ['OK'],
                            title: 'Current Directories',
                            message: `Current directory settings:\n\nDownloads: ${downloadDirectory}\nResearch: ${researchDirectory}`
                        });
                    }
                },
                { type: 'separator' },
                { role: 'quit' }
            ]
        },
        {
            label: 'Edit',
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'delete' },
                { type: 'separator' },
                { role: 'selectAll' }
            ]
        }
    ];

    Menu.setApplicationMenu(Menu.buildFromTemplate(menuTemplate));
};

// Docker service management
const startDockerService = async () => {
    return new Promise((resolve, reject) => {
        // Set environment variables for Docker
        const env = {
            ...process.env,
            OPENAI_API_KEY: 'PLACEHOLDER',
            DOWNLOADS_DIR: downloadDirectory || path.join(app.getPath('downloads')),
            RESEARCH_DIR: researchDirectory || path.join(app.getPath('documents'), 'Research')
        };

        console.log('Starting Docker container with env:', {
            DOWNLOADS_DIR: env.DOWNLOADS_DIR,
            RESEARCH_DIR: env.RESEARCH_DIR
        });

        // Check if Docker is running first
        exec('docker info', async (error) => {
            if (error) {
                const response = await dialog.showMessageBox(mainWindow, {
                    type: 'error',
                    buttons: ['OK'],
                    title: 'Docker Not Running',
                    message: 'Docker is not running. Please start Docker Desktop and try again.',
                    detail: 'This application requires Docker to be installed and running. Please make sure Docker Desktop is started before launching the app.'
                });
                reject(new Error('Docker not running'));
                return;
            }

            // Get the absolute path to docker-compose.yml
            const appPath = app.isPackaged 
                ? path.dirname(app.getPath('exe'))
                : __dirname;
            
            const dockerComposePath = path.join(appPath, 'docker-compose.yml');
            console.log('Docker compose path:', dockerComposePath);

            // Check if docker-compose file exists
            if (!fs.existsSync(dockerComposePath)) {
                const response = await dialog.showMessageBox(mainWindow, {
                    type: 'error',
                    buttons: ['OK'],
                    title: 'Configuration Missing',
                    message: 'Docker configuration file not found.',
                    detail: `Could not find docker-compose.yml at: ${dockerComposePath}`
                });
                reject(new Error('Docker compose file not found'));
                return;
            }

            exec('docker-compose up -d', {
                cwd: appPath,
                env: env
            }, async (error, stdout, stderr) => {
                if (error) {
                    console.error('Error starting Docker:', error);
                    console.error('Docker stdout:', stdout);
                    console.error('Docker stderr:', stderr);

                    const response = await dialog.showMessageBox(mainWindow, {
                        type: 'error',
                        buttons: ['View Error Details', 'OK'],
                        title: 'Docker Error',
                        message: 'Failed to start Docker container.',
                        detail: 'There was an error starting the Docker container. Would you like to see the error details?'
                    });

                    if (response.response === 0) {
                        await dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            buttons: ['OK'],
                            title: 'Docker Error Details',
                            message: 'Docker Error Details:',
                            detail: `Error: ${error}\n\nStdout: ${stdout}\n\nStderr: ${stderr}`
                        });
                    }

                    reject(error);
                    return;
                }

                console.log('Docker service started, waiting for API...');
                console.log('Docker stdout:', stdout);
                
                // Wait for API to be ready with longer timeout
                let retries = 20;
                while (retries > 0) {
                    try {
                        console.log(`Attempting to connect to API (${retries} attempts remaining)...`);
                        const response = await fetch('http://localhost:8000/');
                        if (response.ok) {
                            console.log('API is ready');
                            resolve();
                            return;
                        }
                        console.log('API not ready, status:', response.status);
                    } catch (error) {
                        console.log('API not ready, error:', error.message);
                    }
                    await new Promise(r => setTimeout(r, 1000));
                    retries--;
                }
                
                const timeoutResponse = await dialog.showMessageBox(mainWindow, {
                    type: 'error',
                    buttons: ['OK'],
                    title: 'Connection Timeout',
                    message: 'Failed to connect to the API server.',
                    detail: 'The Docker container started but the API server is not responding. Please check the Docker logs for more information.'
                });
                
                reject(new Error('API failed to start within timeout'));
            });
        });
    });
};

const stopDockerService = async () => {
    return new Promise((resolve, reject) => {
        const appPath = app.isPackaged 
            ? path.dirname(app.getPath('exe'))
            : __dirname;

        exec('docker-compose down', {
            cwd: appPath
        }, (error, stdout, stderr) => {
            if (error) {
                console.error('Error stopping Docker:', error);
                reject(error);
                return;
            }
            console.log('Docker service stopped');
            resolve();
        });
    });
};

const createWindow = async () => {
    // Load settings first
    if (!loadSettings()) {
        await setupDirectories();
    }

    // Start Docker service
    try {
        await startDockerService();
    } catch (error) {
        await dialog.showMessageBox({
            type: 'error',
            title: 'Error',
            message: 'Failed to start the API service. Please make sure Docker is running and try again.'
        });
        app.quit();
        return;
    }

    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    // Load the FastAPI server URL directly
    mainWindow.loadURL('http://localhost:8000');

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    createMenu();
};

app.whenReady().then(createWindow);

app.on('window-all-closed', async () => {
    await stopDockerService();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});

// Clean up Docker when the app is quitting
app.on('before-quit', async (event) => {
    event.preventDefault();
    await stopDockerService();
    app.exit();
});
