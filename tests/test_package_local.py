from docguard_rag import DocGuard


def test_installed_style_local_api(tmp_path):
    source = tmp_path / "document.txt"
    source.write_text(
        "系统通过 /v1 接口提供服务。系统能够被直接掉用。",
        encoding="utf-8",
    )

    with DocGuard(
        database_url=f"sqlite:///{tmp_path / 'local.db'}",
        upload_dir=tmp_path / "uploads",
    ) as guard:
        document = guard.documents.upload(source, enable_review=True)
        assert document.status == "READY"
        assert document.review_id is not None

        result = guard.query(
            "接口版本",
            document_ids=[document.document_id],
        )
        assert result.sources
        assert "/v1" in result.sources[0].text

        issues = guard.reviews.list_issues(document.review_id)
        assert any(issue.issue_type == "TYPO" for issue in issues.items)

        issue = issues.items[0]
        updated = guard.issues.update(
            issue.issue_id,
            status="ACCEPTED",
            comment="已确认",
        )
        assert updated.status == "ACCEPTED"

        job = guard.jobs.get(document.review_id)
        assert job.status == "COMPLETED"
        assert job.progress == 100
