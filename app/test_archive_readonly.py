"""Read-only originals must archive once and leave the watch folder."""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

from naming_utils import archive_readonly_source, clear_readonly


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def test_windows_readonly_delete_is_winerror_5() -> None:
    if os.name != "nt":
        return
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "locked.jpg"
        src.write_bytes(b"readonly-bytes")
        _readonly(src)
        try:
            os.remove(src)
        except PermissionError as e:
            assert getattr(e, "winerror", None) == 5
        else:
            raise AssertionError("readonly delete should raise WinError 5")
        clear_readonly(str(src))
        os.remove(src)
        assert not src.exists()


def test_archive_removes_readonly_and_does_not_duplicate() -> None:
    payload = b"same-bytes-for-archive" * 20
    with tempfile.TemporaryDirectory() as watch, tempfile.TemporaryDirectory() as archive:
        src = Path(watch) / "file-kaMAfETiyTDl7BAKY4mYGANc.jpg"
        src.write_bytes(payload)
        _readonly(src)
        dest, err = archive_readonly_source(str(src), archive)
        assert err is None, err
        assert dest == str(Path(archive) / src.name)
        assert not src.exists()
        assert Path(dest).read_bytes() == payload

        src.write_bytes(payload)
        _readonly(src)
        dest2, err2 = archive_readonly_source(str(src), archive)
        assert err2 is None, err2
        assert dest2 == dest
        assert not src.exists()
        archived = list(Path(archive).iterdir())
        assert len(archived) == 1, [p.name for p in archived]


def test_different_size_keeps_a_second_copy() -> None:
    with tempfile.TemporaryDirectory() as watch, tempfile.TemporaryDirectory() as archive:
        src = Path(watch) / "clip.jpg"
        src.write_bytes(b"aaaa")
        dest, err = archive_readonly_source(str(src), archive)
        assert err is None and dest
        src.write_bytes(b"bbbb-different")
        dest2, err2 = archive_readonly_source(str(src), archive)
        assert err2 is None, err2
        assert dest2 != dest
        assert not src.exists()
        assert len(list(Path(archive).iterdir())) == 2


def test_cross_volume_readonly_move_would_fail_without_clear() -> None:
    """shutil.move across drives copies, then unlink raises WinError 5."""
    if os.name != "nt":
        return
    dest_root = Path(r"D:\OUTPUTS")
    if not dest_root.is_dir():
        return
    scratch = dest_root / "_archive_readonly_test"
    if scratch.exists():
        for p in scratch.rglob("*"):
            if p.is_file():
                clear_readonly(str(p))
        shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    try:
        with tempfile.TemporaryDirectory() as watch:
            src = Path(watch) / "cross.jpg"
            src.write_bytes(b"cross-volume")
            _readonly(src)
            target = scratch / src.name
            try:
                shutil.move(str(src), str(target))
            except PermissionError as e:
                assert getattr(e, "winerror", None) == 5
                assert src.exists()
                assert target.is_file()
            else:
                raise AssertionError("cross-volume shutil.move of a readonly file should fail")
            clear_readonly(str(target))
            target.unlink()
            dest, err = archive_readonly_source(str(src), str(scratch))
            assert err is None, err
            assert not src.exists()
            assert Path(dest).read_bytes() == b"cross-volume"
    finally:
        if scratch.exists():
            for p in scratch.rglob("*"):
                if p.is_file():
                    clear_readonly(str(p))
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    test_windows_readonly_delete_is_winerror_5()
    test_archive_removes_readonly_and_does_not_duplicate()
    test_different_size_keeps_a_second_copy()
    test_cross_volume_readonly_move_would_fail_without_clear()
    print("archive readonly tests ok")
