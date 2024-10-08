# Changelog

## [4.4.0] - 2024-08-10

### Added

- Users can customize Sqlite database name

## [4.3.2] - 2024-06-17

### Added

- Add SingleNDNumpyArray to save high dimensional array
- Better error message for duplicated actors' parameters
- Support primitive dictionary in actor_state

### Changed

- Use serde.csv in Cache.file.csv
- Update `python-slugify`

## [4.3.1] - 2024-04-21

### Fixed

- Fix itemgetter of numpy data models
- Fix type hints error in Python 3.9

## [4.3.0] - 2024-03-26

### Added

- Add FS cli to import/export data
- Retrieve item from SingleNumpyArray with `__getitem__`.

### Fixed

- Update type hints of `get_classpath` to support both classes & functions
- Update `hugedict` to the latest version `2.12.0` that fixed various bugs
- Don't create the folders twice in `FS.export_fs`
- Fix bug that FS does not work when we delete the lastest file/folder.
