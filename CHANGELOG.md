# Changelog

## [4.3.0] - 2024-03-26

### Added

- Add FS cli to import/export data
- Retrieve item from SingleNumpyArray with `__getitem__`.

### Fixed

- Update type hints of `get_classpath` to support both classes & functions
- Update `hugedict` to the latest version `2.12.0` that fixed various bugs
- Don't create the folders twice in `FS.export_fs`
- Fix bug that FS does not work when we delete the lastest file/folder.
