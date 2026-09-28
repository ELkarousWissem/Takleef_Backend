from django.urls import path
from mobicrowd.authentication.login import LoginView
from mobicrowd.authentication.registrationViews import ApprovedRequestersListAPIView, LogoutAPIView, \
    ConfirmEmailAPIView, RequesterApprovalAPIView, \
    RequesterRejectionAPIView, ChangePasswordAPIView, ForgetPasswordView, PasswordResetConfirmView, \
    ActiveWorkersListAPIView, ApprovedRequestersListcountAPIView
from mobicrowd.authentication.session_refresh import SessionBoundTokenRefreshView
from mobicrowd.authentication.takeover_views import VerifyTakeoverOTPAPIView, PendingTakeoverChallengesAPIView, \
    DecideTakeoverChallengeAPIView, ClaimApprovedTakeoverAPIView
from mobicrowd.authentication.user_registration import RegisterRequesterAPIView, RegisterAdminAPIView, \
    RegisterWorkerAPIView, VerifyOTPAPIView, ResendOTPAPIView, RequestRequesterAccessAPIView
from mobicrowd.tasks import \
    DecodeFromEmbeddingView, ServerTimeView, \
    DecodeFromEmbeddingsBatchView
from mobicrowd.video_tasks import RelevanceFromFramesFolderView, RedundancyCheckVideoWithMetadataView, \
    ProcessVideoBlurAndUploadView
from mobicrowd.views.apis.admin_broadcast import BroadcastView, BroadcastTestView
from mobicrowd.views.apis.description_enh.views import  TextEnhExtractAPIView, \
    TextEnhProcessAPIView
from mobicrowd.views.apis.geocode import GeocodeSearchView, GeocodeReverseView
from mobicrowd.views.apis.home_event_views import PublicRequesterHomeEventsView, PublicContributorHomeEventsView, \
    OrganizationRequesterHomeEventsView, OrganizationContributorHomeEventsView
from mobicrowd.views.apis.insert_scraped_data import upload_devices_file, DeviceListView
from mobicrowd.views.apis.notifications import DeviceTokenViewSet, NotificationViewSet
from mobicrowd.views.apis.organization import (
    OrganizationCreateAPIView,
    RequesterEmailListAPIView,
    OrganizationListAPIView,
    OrganizationDeleteAPIView,
    OrganizationUpdateAPIView,
    RepresentativeEmailCheckAPIView,
    RoleRepresentativeCheckAPIView,
    MyRepresentativeOrganizationAPIView,
    OrganizationActivateAPIView,
    MyOrganizationsAPIView,
    OrganizationDetailAPIView,
    OrganizationMembersAPIView,
    OrganizationLicenseSummaryAPIView,
    OrganizationInviteMembersAPIView,
    OrganizationOverviewAPIView,
    OrganizationInvitationCancelAPIView,
    OrganizationInvitationsAPIView,
    OrganizationInvitationAcceptAPIView,
    OrganizationMembershipCreateAPIView,
    OrganizationInvitationDetailByTokenAPIView,
    OrganizationInvitationDeclineAPIView,
    MyRequesterOrganizationsAPIView, OrganizationInvitationUpdateAPIView,
)
from mobicrowd.views.apis.textual_pipeline.views import TextualRelevanceAPIView, TextualRedundancyAPIView
from mobicrowd.views.apis.translate.views import TranslateAPIView, DetectLanguageAPIView
from mobicrowd.views.apis.workshop_apis.recaptured_check.views import RecapturedCheckAPIView
from mobicrowd.views.apis.workshop_apis.story_teller.views import StoryTellerFollowUpAPIView, StoryTellerAPIView
from mobicrowd.views.apis.submission_flow.ocr_extract import ExtractTextApprovedPhotosAPIView
from mobicrowd.views.apis.submission_flow.views import KickoffSubmissionProcessingView, UploadOriginalAndContinueView, \
    KickoffVideoSubmissionProcessingView, UploadOriginalVideoAndContinueView
from mobicrowd.views.apis.workshop_apis.textual_safety_check.views import TextualSafetyAPIView
from mobicrowd.views.apis.users import UserRetrieveAPIView, ListPendingRequestersView, RequesterRetrieveAPIView, \
    ApprovedRequestersNamesView, RequesterLocationView, ProfilePhotoAPIView, UserPhoneView, UserEmailView
from mobicrowd.views.apis.event_views import DeleteWorkerJoinView, AvailableEventsForWorkerView, TotalUsersAPIView, \
    TotalEventsCreatedTodayAPIView, RejectJoinRequestView, \
    PendingInvitationsView, EventRetrieveByIdView, JoinEventView, \
    EventListView, EventRetrieveView, EventCreateView, EventUpdateView, EventDeleteView, ApproveJoinRequestView, \
    UserInfoAPIView, UpcomingEventsListAPIView, PastEventsListAPIView, EventListFilterView, \
    WorkerJoinedEventsUpcomingView, WorkerJoinedEventsPastView, WorkerJoinedEventsByIdView, RequesterEventsPastView, \
    RequesterEventsAllView, RequesterEventsUpcomingView, PendingWorkersCountAPIView, \
    WorkerStatusView, WorkerEventDetailsView, EventProgressView, EventSubmissionReportsAPIView, \
    OrganizationEventListView, OrganizationEventCreateView, MyOrganizationEventsListView, \
    OrganizationAvailableEventsForWorkerView, OrganizationEventsListAPIView, \
    OrganizationUpcomingEventsListAPIView, OrganizationPastEventsListAPIView
from mobicrowd.views.apis.submission_views import AcceptedSubmissionCountView, \
    SubmissionStatusUpdateAPIView, DownloadImageUrlView, DownloadEventZipView, ALLAcceptedSubmissionCountView, \
    ALLRejectedSubmissionCountView, RefusedSubmissionCountView, AllSubmissionCountView, SubmissionReportListCreate, \
    SubmissionReportListAll, ApprovedSubmissionsByWorkerView, WorkerEventEarningsView, SubmissionReportCreateAPIView, \
    AddSubmissionMessageAPIView, MarkSubmissionMessageReadAPIView, DownloadSubmissionReportUrlView
from mobicrowd.views.apis.submission_views import SubmissionCountView, SubmissionCreateView, SubmissionStatusView, \
    InProcessingSubmissionsByEventView, InProcessingSubmissionsByWorkerView, \
    ApprovedSubmissionsByEventView, WorkerEventInfoAPIView
from mobicrowd.views.apis.statistic_views import RequesterEventStatisticsView, WorkersStatisticsDashboardView, \
    AllWorkersStatisticsView, CurrentEventsWithContributorsView, AdminDashboardView, \
    OrganizationRequesterDashboardView, OrganizationContributorDashboardView, OrganizationAdminDashboardView
from mobicrowd.views.apis.workshop_apis.annotation.views import OpenRouterAnnotationAPIView
from mobicrowd.views.apis.workshop_apis.background_removal.views import BackgroundRemovalAPIView
from mobicrowd.views.apis.workshop_apis.image_segmentation.views import ImageSegmentationAPIView
from mobicrowd.views.apis.workshop_apis.views import WorkshopListCreateAPIView, WorkshopDetailAPIView, \
    WorkshopMediaListAddAPIView, WorkshopMediaDetailAPIView, WorkshopActionResultListCreateAPIView

device_list = DeviceTokenViewSet.as_view({"get": "list", "post": "create"})
device_detail = DeviceTokenViewSet.as_view({"delete": "destroy"})

notif_list = NotificationViewSet.as_view({"get": "list"})
notif_unread = NotificationViewSet.as_view({"get": "unread_count"})
notif_mark_read = NotificationViewSet.as_view({"post": "mark_read"})
notif_mark_all = NotificationViewSet.as_view({"post": "mark_all_read"})

urlpatterns = [
path(
    'token/refresh',
    SessionBoundTokenRefreshView.as_view(),
    name='token-refresh'
),
    path("auth/takeover/verify-otp/", VerifyTakeoverOTPAPIView.as_view()),
    path("auth/takeover/pending/", PendingTakeoverChallengesAPIView.as_view()),
    path("auth/takeover/decide/", DecideTakeoverChallengeAPIView.as_view()),
    path("auth/takeover/claim-approved/", ClaimApprovedTakeoverAPIView.as_view()),
    path(
        "users/profile-photo/",
        ProfilePhotoAPIView.as_view(),
        name="profile-photo"
    ),
    path('users/<int:id>', UserRetrieveAPIView.as_view(), name='user-detail'),
    path('requester/<int:user_id>', RequesterRetrieveAPIView.as_view(), name='user-detail'),
    path('register_requester', RegisterRequesterAPIView.as_view(), name='register_requester'),
    path('register_worker', RegisterWorkerAPIView.as_view(), name='register_worker'),
    path('verify-otp', VerifyOTPAPIView.as_view(), name='verify-otp'),
    path('resend-otp', ResendOTPAPIView.as_view(), name='resend-otp'),
    #path('register_admin', RegisterAdminAPIView.as_view(), name='register_admin'),
    path('login', LoginView.as_view(), name='login'),
    path('account_confirm_email/<int:pk>/<str:token>', ConfirmEmailAPIView.as_view(), name='account_confirm_email'),
    path('requester/<str:email>/approve-account', RequesterApprovalAPIView.as_view(), name='approve-requester-account'),
    path('requester/<str:email>/reject-account', RequesterRejectionAPIView.as_view(), name='reject-requester-account'),
    path('requesters/approved', ApprovedRequestersListAPIView.as_view(), name='approved-requesters-list'),
    path('user/change-password', ChangePasswordAPIView.as_view(), name='change-password'),
    path('forget-password/<str:email>', ForgetPasswordView.as_view(), name='forget_password'),
    path(
        'set-new-password/<uuid:selector>/<str:token>',
        PasswordResetConfirmView.as_view(),
        name='set-new-password'
    ),
    path('logout', LogoutAPIView.as_view(), name='logout'),
    path('user/info', UserInfoAPIView.as_view(), name='user-info'),
    path("user/change-phone", UserPhoneView.as_view(), name="user-change-phone"),
    path("user/change-location", RequesterLocationView.as_view(), name="requester-change-location"),
    path("user/change-email", UserEmailView.as_view(), name="user-change-email"),
    path('PendingRequester', ListPendingRequestersView.as_view(), name='listpendingRequester'),
    path('pending-invitations', PendingInvitationsView.as_view(), name='pending-invitations'),
    path('requesters/names', ApprovedRequestersNamesView.as_view(), name='approved-requesters'),
    path("requester/<int:user_id>/location", RequesterLocationView.as_view(), name="requester-location"),
    path("geocode/search", GeocodeSearchView.as_view()),
    path("geocode/reverse", GeocodeReverseView.as_view()),

    # decoder model api
    # path('embedding/' , receive_embedding , name='receive_embedding' ),

    ##### events apis #############
    path('events/<int:pk>', EventRetrieveByIdView.as_view(), name='event-retrieve-by-id'),
    path('events/requester/all', RequesterEventsAllView.as_view(), name='requester-events'),
    path('events/requester/upcoming', RequesterEventsUpcomingView.as_view(), name='requester-events'),
    path('events/requester/past', RequesterEventsPastView.as_view(), name='requester-events'),
    path('events', EventListView.as_view(), name='event-list'),
    path('events/create', EventCreateView.as_view(), name='event-create'),
    path('events/<int:pk>', EventRetrieveView.as_view(), name='event-detail'),
    path('events/<int:pk>/update', EventUpdateView.as_view(), name='event-update'),
    path('events/<int:pk>/delete', EventDeleteView.as_view(), name='event-delete'),
    path('events/join/<int:event_id>', JoinEventView.as_view(), name='join-event'),
    path('events/<int:event_id>/worker_status', WorkerStatusView.as_view(), name='worker-status'),
    path('events/<int:event_id>/worker_event_details', WorkerEventDetailsView.as_view(), name='worker-details'),
    path('events/<int:event_id>/event_details', EventProgressView.as_view(), name='worker-details'),

    path('events/approve/<int:event_worker_id>', ApproveJoinRequestView.as_view(), name='approve-join-request'),
    path('events/refuse/<int:event_worker_id>', RejectJoinRequestView.as_view(), name='approve-join-request'),
    path('worker/<int:worker_id>/events/joined/all', WorkerJoinedEventsByIdView.as_view(), name='worker-joined-events'),
    path('worker/<int:worker_id>/events/joined/upcoming', WorkerJoinedEventsUpcomingView.as_view(),
         name='worker-joined-events'),
    path('worker/<int:worker_id>/events/joined/past', WorkerJoinedEventsPastView.as_view(),
         name='worker-joined-events'),
    path('events/past', PastEventsListAPIView.as_view(), name='past-events-list'),
    path('events/upcoming', UpcomingEventsListAPIView.as_view(), name='upcoming-events-list'),
    path('events/filter', EventListFilterView.as_view(), name='event-list'),
    path('events/<int:event_id>/cancel-join/<int:user_id>/', DeleteWorkerJoinView.as_view(), name='cancel-join'),
    path('active-workers', ActiveWorkersListAPIView.as_view(), name='active-workers-list'),
    path('requester-count', ApprovedRequestersListcountAPIView.as_view(), name='not-approved-requesters-count'),
    path('pending-workers-count', PendingWorkersCountAPIView.as_view(), name='pending-workers-count'),
    path('events/available-for-worker', AvailableEventsForWorkerView.as_view(), name='available-events-for-worker'),
    #### insert scraped data ###
    path('upload-devices', upload_devices_file, name='upload_devices_file'),
    path('devices-spec-filter', DeviceListView.as_view(), name='device-list'),

    ### submission #############
    path('submission-count-worker/<int:event_id>/<int:worker_id>', SubmissionCountView.as_view(),
         name='submission-count'),
    path('submission-count-all/<int:event_id>', AllSubmissionCountView.as_view(), name='submission-count'),
    path('submission-count-accepted/<int:event_id>/<int:worker_id>', AcceptedSubmissionCountView.as_view(),
         name='submission-count'),
    path('submission-count-rejected/<int:event_id>/<int:worker_id>', RefusedSubmissionCountView.as_view(),
         name='submission-count'),
    path('all-accepted-submissions/<int:event_id>', ALLAcceptedSubmissionCountView.as_view(), name='submission-count'),
    path('all-rejected-submissions/<int:event_id>', ALLRejectedSubmissionCountView.as_view(), name='submission-count'),
    path('submissions/create', SubmissionCreateView.as_view(), name='submission-create'),
    path('submissions/status/<int:submission_id>', SubmissionStatusView.as_view(), name='submission-status'),
    path('events/<int:event_id>/worker-info', WorkerEventInfoAPIView.as_view(), name='worker-event-info'),
    path('events/<int:event_id>/submissions-in-processing', InProcessingSubmissionsByEventView.as_view(),
         name='submissions_by_event'),
    path('events/<int:event_id>/submissions-approved', ApprovedSubmissionsByEventView.as_view(),
         name='approved-submissions_by_event'),
    path('events/<int:event_id>/worker/<int:worker_id>/submissions-in-processing',
         InProcessingSubmissionsByWorkerView.as_view(), name='submissions_by_event'),
    path('events/<int:event_id>/worker/<int:worker_id>/submissions-approved', ApprovedSubmissionsByWorkerView.as_view(),
         name='submissions_by_event'),
    # path('photos/<int:event_id>/<int:user_id>/<int:submission_id>/upload', SaveBase64ImageToS3View.as_view(), name='save_base64_image_to_s3'),
    path('server-time', ServerTimeView.as_view(), name='server-time'),
    path('create-report', SubmissionReportListCreate.as_view(), name='report-list-create'),
    path('reports/all', SubmissionReportListAll.as_view(), name='report-list-all'),
    path('events/<int:event_id>/workers/<int:worker_id>/earnings', WorkerEventEarningsView.as_view(),
         name='worker_event_earnings'),
    path("photos/extract-text", ExtractTextApprovedPhotosAPIView.as_view(), name="ocr-extract"),

    # path('quality-assessment/<int:submission_id>', QualityAssessmentView.as_view(), name='quality-assessment'),

    ###############takleef 2.0################################

    # path('image-quality/<int:submission_id>/<int:user_id>', CheckImageQualityView.as_view(), name='check-image-quality'),
    # path("relevance_check", RelevanceCheckView.as_view(), name="relevance_check"),
    # path("redundancy_check", RedundancyCheckWithMetadataView.as_view(), name="redundancy_check"),
    # path('blur-and-upload', BlurFacesAndUploadView.as_view(), name='blur_and_upload'),
    path("decode/<int:submission_id>", DecodeFromEmbeddingView.as_view(), name="decode-from-file-secure"),
    # Legacy route kept temporarily; the view verifies user_id == request.user.id.
    path("decode/<int:submission_id>/<int:user_id>", DecodeFromEmbeddingView.as_view(), name="decode-from-file"),
    path("submission/<int:pk>/status", SubmissionStatusUpdateAPIView.as_view(), name='submission-status-update'),
    path('download-url/<int:photo_id>', DownloadImageUrlView.as_view(), name='download-url'),
    path("submission-reports/<int:report_id>/download-url", DownloadSubmissionReportUrlView.as_view()),
    path('download-zip/event/<int:event_id>', DownloadEventZipView.as_view(),
         name='event-zip'),
    path('submission/add-message', AddSubmissionMessageAPIView.as_view(), name='add-submission-message'),
    path('submission/mark-message-read', MarkSubmissionMessageReadAPIView.as_view(),
         name='mark-submission-message-read'),
    path('event/<int:event_id>/reports', EventSubmissionReportsAPIView.as_view(), name='event-submission-reports'),
    # video ###########################################################"
    path("decode-embeddings/<int:submission_id>", DecodeFromEmbeddingsBatchView.as_view(),
         name="decode-embeddings-secure"),
    # Legacy route kept temporarily; authorization is still derived from request.user.
    path("decode-embeddings/<int:submission_id>/<int:user_id>", DecodeFromEmbeddingsBatchView.as_view(),
         name="decode-embeddings"),

    #################################video qwen#####################################""
    path(
        "submissions/video/kickoff/<int:submission_id>",
        KickoffVideoSubmissionProcessingView.as_view(),
        name="submission-video-kickoff",
    ),
    path(
        "submissions/video/upload-original/<int:submission_id>",
        UploadOriginalVideoAndContinueView.as_view(),
        name="submission-video-upload-original",
    ),
    ############################celery workflow#######################################################"""
    path("submissions/<int:submission_id>/kickoff", KickoffSubmissionProcessingView.as_view()),
    path("submissions/<int:submission_id>/original", UploadOriginalAndContinueView.as_view()),

    path('total-events-today', TotalEventsCreatedTodayAPIView.as_view(), name='total-events-today'),
    path('total-users', TotalUsersAPIView.as_view(), name='total-users'),
    path('admin-dashboard', AdminDashboardView.as_view(), name='admin-dashboard'),
    path('workers-statistics', AllWorkersStatisticsView.as_view(), name='all-workers-statistics'),
    path('current-events-with-contributors', CurrentEventsWithContributorsView.as_view(),
         name='current-events-with-contributors'),
    path('requester-event-statistics', RequesterEventStatisticsView.as_view(), name='requester_event_statistics'),
    path('worker-event-statistics/<int:worker_id>', WorkersStatisticsDashboardView.as_view(),
         name='worker_event_static'),

    path("devices/", device_list, name="device-token-list"),
    path("devices/<int:pk>/", device_detail, name="device-token-detail"),
    path("notifications/", notif_list, name="notification-list"),
    path("notifications/unread_count/", notif_unread, name="notification-unread"),
    path("notifications/<int:pk>/mark_read/", notif_mark_read, name="notification-mark-read"),
    path("notifications/mark_all_read/", notif_mark_all, name="notification-mark-all"),
    path("admin/broadcast", BroadcastView.as_view()),
    path("admin/broadcast/test/", BroadcastTestView.as_view()),  # <-- NEW

    path('report', SubmissionReportCreateAPIView.as_view(), name='submission-report-create'),

    ################################################organization apis #########################################
    # Keep the old no-slash route for existing frontend calls, and add
    # the slash route used by the organization feature branch.
    path('organizations/create/', OrganizationCreateAPIView.as_view(), name='organization-create-slash'),
    path('requesters/emails/', RequesterEmailListAPIView.as_view(), name='requester-email-list'),
    path('organizations/', OrganizationListAPIView.as_view(), name='organization-list'),
    path('organizations/delete/<int:org_id>/', OrganizationDeleteAPIView.as_view(), name='organization-delete'),
    path('organizations/update/<int:org_id>/', OrganizationUpdateAPIView.as_view(), name='organization-update'),
    path('representative-email-check/', RepresentativeEmailCheckAPIView.as_view(), name='representative-email-check'),
    path('auth/rep/', RoleRepresentativeCheckAPIView.as_view(), name='auth-representative-check'),
    path(
        'organizations/my-representative-organization/',
        MyRepresentativeOrganizationAPIView.as_view(),
        name='my-representative-organization',
    ),

    # Organization event routes. These depend only on event_views.py and are kept
    # separate from organization dashboard/statistics routes.
    path('organizations/events/', OrganizationEventsListAPIView.as_view(), name='organizations-events'),
    path(
        'organizations/events/upcoming/',
        OrganizationUpcomingEventsListAPIView.as_view(),
        name='organizations-events-upcoming',
    ),
    path(
        'organizations/events/past/',
        OrganizationPastEventsListAPIView.as_view(),
        name='organizations-events-past',
    ),
    path(
        'organizations/<int:org_id>/my-events/',
        MyOrganizationEventsListView.as_view(),
        name='my-organization-events',
    ),
    path(
        'organizations/<int:org_id>/available-events/',
        OrganizationAvailableEventsForWorkerView.as_view(),
        name='organization-available-events-for-worker',
    ),
    path(
        'organizations/<int:org_id>/events/',
        OrganizationEventListView.as_view(),
        name='organization-events',
    ),
    path(
        'organizations/<int:org_id>/events/create/',
        OrganizationEventCreateView.as_view(),
        name='organization-event-create',
    ),
    path('organizations/activate/<int:org_id>/', OrganizationActivateAPIView.as_view(), name='organization-activate'),
    path('organizations/my-organizations/', MyOrganizationsAPIView.as_view(), name='my-organizations'),
    # Organization dashboard/statistics routes. Keep them before the generic
    # organizations/<int:org_id>/ detail route.
    path(
        'organizations/<int:org_id>/requester-dashboard/',
        OrganizationRequesterDashboardView.as_view(),
        name='organization-requester-dashboard',
    ),
    path(
        'organizations/<int:org_id>/contributor-dashboard/',
        OrganizationContributorDashboardView.as_view(),
        name='organization-contributor-dashboard',
    ),
    path(
        'organizations/<int:org_id>/admin-dashboard/',
        OrganizationAdminDashboardView.as_view(),
        name='organization-admin-dashboard',
    ),
    path('organizations/<int:org_id>/', OrganizationDetailAPIView.as_view(), name='organization-detail'),
    path('organizations/<int:org_id>/members/', OrganizationMembersAPIView.as_view(), name='organization-members'),
    path(
        'organizations/<int:org_id>/license-summary/',
        OrganizationLicenseSummaryAPIView.as_view(),
        name='organization-license-summary',
    ),
    path('organizations/<int:org_id>/invite/', OrganizationInviteMembersAPIView.as_view(), name='organization-invite'),
    path('organizations/<int:org_id>/overview/', OrganizationOverviewAPIView.as_view(), name='organization-overview'),
    path(
        'organization-invitations/<int:invitation_id>/cancel/',
        OrganizationInvitationCancelAPIView.as_view(),
        name='organization-invitation-cancel',
    ),
    path(
        'organization-invitations/<int:invitation_id>/update/',
        OrganizationInvitationUpdateAPIView.as_view(),
        name='organization-invitation-update',
    ),
    path(
        'organizations/<int:org_id>/invitations/',
        OrganizationInvitationsAPIView.as_view(),
        name='organization-invitations',
    ),
    path(
        'organizations/invitations/accept/',
        OrganizationInvitationAcceptAPIView.as_view(),
        name='organization-invitation-accept',
    ),
    path(
        'organizations/<int:org_id>/memberships/',
        OrganizationMembershipCreateAPIView.as_view(),
        name='organization-membership-create',
    ),
    path(
        'organization-invitations/detail/',
        OrganizationInvitationDetailByTokenAPIView.as_view(),
        name='organization-invitation-detail-by-token',
    ),
    path(
        'organization-invitations/decline/',
        OrganizationInvitationDeclineAPIView.as_view(),
        name='organization-invitation-decline',
    ),
    path('my-requester-organizations', MyRequesterOrganizationsAPIView.as_view(), name='my-requester-organizations'),
    path('requester-access-request/', RequestRequesterAccessAPIView.as_view(), name='requester-access-request'),


######################################### home #######################################

    path(
        "home/events/public/requester/",
        PublicRequesterHomeEventsView.as_view(),
        name="home-events-public-requester",
    ),
    path(
        "home/events/public/contributor/",
        PublicContributorHomeEventsView.as_view(),
        name="home-events-public-contributor",
    ),
    path(
        "home/events/organizations/<int:organization_id>/requester/",
        OrganizationRequesterHomeEventsView.as_view(),
        name="home-events-organization-requester",
    ),
    path(
        "home/events/organizations/<int:organization_id>/contributor/",
        OrganizationContributorHomeEventsView.as_view(),
        name="home-events-organization-contributor",
    ),
################################workshop###############################################""

    path(
        "annotate",
        OpenRouterAnnotationAPIView.as_view(),
        name="openrouter-annotate",
    ),
    path(
        "remove-background",
        BackgroundRemovalAPIView.as_view(),
        name="remove-background",
    ),
    path(
        "segment-image",
        ImageSegmentationAPIView.as_view(),
        name="segment-image",
    ),
    path(
        "recaptured-check",
        RecapturedCheckAPIView.as_view(),
        name="recaptured-check",
    ),
    path(
        "story-teller",
        StoryTellerAPIView.as_view(),
        name="story-teller",
    ),
    path(
        "story-teller/follow-up",
        StoryTellerFollowUpAPIView.as_view(),
        name="story-teller-follow-up",
    ),
    path(
        "textual-safety",
        TextualSafetyAPIView.as_view(),
        name="textual-safety",
    ),
    path('workshops/', WorkshopListCreateAPIView.as_view(), name='workshop-list-create'),
    path('workshops/<int:pk>/', WorkshopDetailAPIView.as_view(), name='workshop-detail'),

    path('workshops/<int:pk>/media/', WorkshopMediaListAddAPIView.as_view(), name='workshop-media-list-add'),
    path('workshops/<int:pk>/media/<int:media_id>/', WorkshopMediaDetailAPIView.as_view(), name='workshop-media-detail'),
path(
    "workshops/<int:pk>/action-results/",
    WorkshopActionResultListCreateAPIView.as_view(),
    name="workshop-action-results",
),
############################################## description enhancement #################################""

    path("extract", TextEnhExtractAPIView.as_view(), name="api-enhance-extract"),
    path("process", TextEnhProcessAPIView.as_view(), name="api-enhance-process"),

############################# textual pipeline ####################################

    path("text-relevance", TextualRelevanceAPIView.as_view(), name="api-textual-relevance"),
    path("text-redundancy", TextualRedundancyAPIView.as_view(), name="api-textual-redundancy"),

################################# translate ##############################
    path("detect-lang", DetectLanguageAPIView.as_view(), name="detect-lang"),
    path("translate", TranslateAPIView.as_view(), name="translate"),
]