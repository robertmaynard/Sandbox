Minimal test case to show an issue with cmake commit: 4959f3413c83


Current patch to fix this issue is:

From c89e11f9398c82e9d65929ed3d2a6872791b0942 Mon Sep 17 00:00:00 2001
From: Robert Maynard <robert.maynard@kitware.com>
Date: Mon, 19 May 2014 16:24:55 -0400
Subject: [PATCH] Unify all paths that custom command uses to be full paths.

---
 Source/cmAddCustomCommandCommand.cxx | 3 ++-
 1 file changed, 2 insertions(+), 1 deletion(-)

diff --git a/Source/cmAddCustomCommandCommand.cxx b/Source/cmAddCustomCommandCommand.cxx
index d5f00ff..417e5be 100644
--- a/Source/cmAddCustomCommandCommand.cxx
+++ b/Source/cmAddCustomCommandCommand.cxx
@@ -162,6 +162,7 @@ bool cmAddCustomCommandCommand
             }
           filename += copy;
           cmSystemTools::ConvertToUnixSlashes(filename);
+          filename = cmSystemTools::CollapseFullPath(filename.c_str());
           break;
         case doing_source:
           // We do not want to convert the argument to SOURCE because
@@ -218,7 +219,7 @@ bool cmAddCustomCommandCommand
            {
            std::string dep = copy;
            cmSystemTools::ConvertToUnixSlashes(dep);
-           depends.push_back(dep);
+           depends.push_back( cmSystemTools::CollapseFullPath(dep.c_str()) );
            }
            break;
          case doing_outputs:
--
1.8.5.2 (Apple Git-48)