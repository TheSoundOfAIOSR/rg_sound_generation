DROP TABLE IF EXISTS user;
DROP TABLE IF EXISTS sample;

CREATE TABLE user (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE sample (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    q_bright INTEGER NOT NULL,
    q_dark INTEGER NOT NULL,
    q_full INTEGER NOT NULL,
    q_hollow INTEGER NOT NULL,
    q_smooth INTEGER NOT NULL,
    q_rough INTEGER NOT NULL,
    q_warm INTEGER NOT NULL,
    q_metallic INTEGER NOT NULL,
    q_clear INTEGER NOT NULL,
    q_muddy INTEGER NOT NULL,
    q_thin INTEGER NOT NULL,
    q_thick INTEGER NOT NULL,
    q_pure INTEGER NOT NULL,
    q_noisy INTEGER NOT NULL,
    q_rich INTEGER NOT NULL,
    q_sparse INTEGER NOT NULL,
    q_harmonic INTEGER NOT NULL,
    q_disharmonic INTEGER NOT NULL,
    q_soft INTEGER NOT NULL,
    q_hard INTEGER NOT NULL,
    description TEXT
);
