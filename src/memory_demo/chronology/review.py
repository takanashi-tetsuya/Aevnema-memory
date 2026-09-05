from __future__ import annotations

from memory_demo.repositories import AssociationRepository, EpisodeRepository, SourceRepository


class TimelineReviewer:
    def __init__(
        self,
        episodes: EpisodeRepository,
        sources: SourceRepository,
        associations: AssociationRepository,
    ):
        self.episodes = episodes
        self.sources = sources
        self.associations = associations

    def export_markdown(self, timeline_scope: str) -> str:
        rows = self.episodes.list_timeline(timeline_scope)
        lines = [f"# Timeline: {timeline_scope}", ""]
        for row in rows:
            order = row["story_order"] if row["story_order"] is not None else "?"
            lines.extend(
                [
                    f"## Episode #{row['id']} · order={order}",
                    "",
                    f"- Source: `{row['source_key']}` segment {row['segment_index']}",
                    f"- Story time: {row['story_time_text'] or '未知'}",
                    f"- Confidence: {row['confidence']}",
                    "",
                    str(row["text"]),
                    "",
                ]
            )
            temporal = [
                edge
                for edge in self.associations.neighbors("episode", int(row["id"]), 100)
                if edge["relation_type"] == "temporal"
            ]
            if temporal:
                lines.append("Temporal associations:")
                lines.append("")
                for edge in temporal:
                    lines.append(
                        f"- #{edge['id']} `{edge['relation_key']}`: {edge['relation_text']}"
                    )
                lines.append("")
        return "\n".join(lines)

