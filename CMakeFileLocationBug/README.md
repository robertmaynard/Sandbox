Minimal test case to show an issue with cmake commit: 4959f3413c83


Current patch to fix this issue is:
From 525eced028a794c892a768c333541832bb289ea1 Mon Sep 17 00:00:00 2001
From: Robert Maynard <robert.maynard@kitware.com>
Date: Mon, 19 May 2014 17:13:23 -0400
Subject: [PATCH] Unify all paths that custom command uses to be full paths.

---
 Source/cmAddCustomCommandCommand.cxx | 14 +++++++++++++-
 1 file changed, 13 insertions(+), 1 deletion(-)

diff --git a/Source/cmAddCustomCommandCommand.cxx b/Source/cmAddCustomCommandCommand.cxx
index d5f00ff..02f9df6 100644
--- a/Source/cmAddCustomCommandCommand.cxx
+++ b/Source/cmAddCustomCommandCommand.cxx
@@ -162,6 +162,10 @@ bool cmAddCustomCommandCommand
             }
           filename += copy;
           cmSystemTools::ConvertToUnixSlashes(filename);
+          if (cmSystemTools::FileIsFullPath(filename.c_str()))
+            {
+            filename = cmSystemTools::CollapseFullPath(filename.c_str());
+            }
           break;
         case doing_source:
           // We do not want to convert the argument to SOURCE because
@@ -197,6 +201,10 @@ bool cmAddCustomCommandCommand
            // explicit dependency.
            std::string dep = copy;
            cmSystemTools::ConvertToUnixSlashes(dep);
+           if (cmSystemTools::FileIsFullPath(dep.c_str()))
+            {
+            dep = cmSystemTools::CollapseFullPath(dep.c_str());
+            }
            depends.push_back(dep);

            // Add the implicit dependency language and file.
@@ -218,7 +226,11 @@ bool cmAddCustomCommandCommand
            {
            std::string dep = copy;
            cmSystemTools::ConvertToUnixSlashes(dep);
-           depends.push_back(dep);
+           if (cmSystemTools::FileIsFullPath(dep.c_str()))
+            {
+            dep = cmSystemTools::CollapseFullPath(dep.c_str());
+            }
+           depends.push_back( dep.c_str() );
            }
            break;
          case doing_outputs:
--
1.8.5.2 (Apple Git-48)