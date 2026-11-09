"""The widget loader supports installed package data and editable checkouts."""

from vectoreyes import mcp_server


def test_packaged_widget_takes_precedence_over_checkout(tmp_path, monkeypatch):
    package = tmp_path / "package"
    asset = package / "assets" / "mcp-app.html"
    asset.parent.mkdir(parents=True)
    asset.write_text("<html>Packaged review — SVG</html>", encoding="utf-8")
    checkout = tmp_path / "checkout.html"
    checkout.write_text("stale checkout", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "files", lambda name: package)
    monkeypatch.setattr(mcp_server, "_WIDGET_HTML_PATH", checkout)

    assert mcp_server.logo_widget() == asset.read_text(encoding="utf-8")


def test_editable_widget_reads_current_build(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "files", lambda name: tmp_path / "package")
    checkout = tmp_path / "checkout.html"
    checkout.write_text("<html>Editable review</html>", encoding="utf-8")
    monkeypatch.setattr(mcp_server, "_WIDGET_HTML_PATH", checkout)

    assert mcp_server.logo_widget() == checkout.read_text(encoding="utf-8")


def test_unbuilt_checkout_explains_required_build(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "files", lambda name: tmp_path / "package")
    monkeypatch.setattr(mcp_server, "_WIDGET_HTML_PATH", tmp_path / "missing.html")

    assert mcp_server.logo_widget() == mcp_server._WIDGET_BUILD_REQUIRED_HTML
