# Claude Code Memory - Image Search Project

## Project Context
This is an image search project using CLIP embeddings, FastAPI backend, Streamlit frontend, and pgvector database for similarity search.

## Recent Work: Database Persistence Migration

### Current Setup
- Migrated from Docker named volumes to bind mounts for better data persistence
- PostgreSQL container now uses bind mount: `./postgres_data:/var/lib/postgresql/data`
- Data persists even with `docker-compose down -v`

### Key Files Modified
- `docker-compose.yml`: Changed pgvector service volume from named volume to bind mount
- `.gitignore`: Added `postgres_data/` to exclude database files from git

### Database Migration Process
1. **Backup Creation**: Created `postgres_backup.tar.gz` using Docker container
   ```bash
   docker exec pgvector-db tar -czf /tmp/postgres_backup.tar.gz -C /var/lib/postgresql/data .
   docker cp pgvector-db:/tmp/postgres_backup.tar.gz .
   ```

2. **Migration Steps for New Machine**:
   ```bash
   # Extract backup
   mkdir postgres_data
   tar -xzf postgres_backup.tar.gz -C postgres_data/
   
   # Set correct permissions (postgres user UID 999)
   sudo chown -R 999:999 postgres_data/
   
   # Start containers
   docker-compose up -d
   ```

### Database Connection Details
- Host: localhost:18152
- Database: clipdb_v2
- User: clipuser
- Password: clippass

### Test Data Verification
The database contains a test table `test_persist` with sample data to verify migration success:
```sql
SELECT * FROM test_persist;
-- Should return: id=1, data='persistence test'
```

### Known Issues & Solutions
- **uvicorn file watcher error**: Added `postgres_data/` to `.gitignore` to prevent permission issues
- **Permission denied on postgres_data**: Normal behavior - directory owned by UID 999 (postgres user)

## Migration Checklist for Home PC
- [ ] Pull latest git changes (includes docker-compose.yml updates)
- [ ] Download postgres_backup.tar.gz from Google Drive  
- [ ] Extract backup to postgres_data/ directory
- [ ] Set permissions: `sudo chown -R 999:999 postgres_data/`
- [ ] Start containers: `docker-compose up -d`
- [ ] Verify test data exists: Connect to database and check `test_persist` table
- [ ] Continue development with persistent data

## Development Notes
- Use `docker-compose down` (without -v) to preserve data during rebuilds
- Only use `docker-compose down -v` when intentionally wiping data
- Backup important data before major changes