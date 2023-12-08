const RELEASE_BRANCH = process.env.RELEASE_BRANCH || "main";
const CHANGELOG_FILE = process.env.CHANGELOG_FILE || "CHANGELOG.md";
const VERSION_FILE = process.env.VERSION_FILE || "src/*/_version.py";

const config = {
	branches: [RELEASE_BRANCH],
	plugins: [
		[
			"@semantic-release/commit-analyzer",
			{
				preset: "conventionalcommits",
			},
		],
		[
			"@semantic-release/release-notes-generator",
			{
				preset: "conventionalcommits",
			},
		],
		[
			"@semantic-release/changelog",
			{
				changelogFile: CHANGELOG_FILE,
				changelogTitle:
					"# Changelog\n\nAll notable changes to this project will be documented in this file. See\n[Conventional Commits](https://conventionalcommits.org) for commit guidelines.",
			},
		],
		[
			"@semantic-release/exec",
			{
				prepareCmd: "poetry version ${nextRelease.version} && poetry build",
			},
		],
		[
			"@google/semantic-release-replace-plugin",
			{
				replacements: [
					{
						files: [VERSION_FILE],
						ignore: ["test/*", "tests/*"],
						from: "__version__ = [\"'].*[\"']",
						to: '__version__ = "${nextRelease.version}"',
					},
				],
			},
		],
		[
			"@semantic-release/github",
			{
				assets: [
					{ path: "dist/*.tar.gz", label: "sdist" },
					{ path: "dist/*.whl", label: "wheel" },
				],
				successComment: false,
				failComment: false,
				releasedLabels: false,
				failTitle: false,
				labels: false,
			},
		],
		[
			"@semantic-release/git",
			{
				assets: ["pyproject.toml", VERSION_FILE, CHANGELOG_FILE],
			},
		],
	],
};

module.exports = config;
